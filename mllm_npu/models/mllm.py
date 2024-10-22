import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import LogitsProcessor, LogitsProcessorList

from mllm_npu.utils import load_zero3_checkpoint
from mllm_npu.constant import BOI_TOKEN, EOI_TOKEN, IMG_TOKEN


def cosine_loss(rec, target):
    target = target / target.norm(dim=-1, keepdim=True)
    rec = rec / rec.norm(dim=-1, keepdim=True)
    rec_loss = (1 - (target * rec).sum(-1)).mean()
    return rec_loss


class AutoImageTokenGenerationProcessor(LogitsProcessor):

    def __init__(self, tokenizer, num_img_gen_tokens=64) -> None:
        super().__init__()
        img_all_token_str = ''.join([BOI_TOKEN] + [
            IMG_TOKEN.format(int(item)) for item in range(num_img_gen_tokens)
        ] + [EOI_TOKEN])
        self.img_ids_list = tokenizer.encode(img_all_token_str,
                                             add_special_tokens=False)

    def __call__(self, input_ids, scores):
        bz = input_ids.shape[0]
        for i in range(bz):
            cur_input_id = input_ids[i, -1].item()
            if cur_input_id in self.img_ids_list[:-1]:

                output_id = self.img_ids_list[
                    self.img_ids_list.index(cur_input_id) + 1]
                scores[i, ..., output_id] = scores[i, ...].max() + 10.
            else:

                scores[i, ...,
                       torch.tensor(self.img_ids_list[1:]).to(
                           dtype=torch.long)] = 0.0

        return scores


class GeneraliazedMultimodalModels(nn.Module):

    def __init__(self,
                 language_model,
                 vision_encoder,
                 projector,
                 freeze_vision_encoder=True,
                 lm_loss_scale=1.0,
                 add_patch_pos=False) -> None:
        super().__init__()
        self.language_model = language_model
        self.vision_encoder = vision_encoder
        self.vision_encoder.requires_grad_(not freeze_vision_encoder)
        self.projector = projector
        self.freeze_vision_encoder = freeze_vision_encoder
        self.lm_loss_scale = lm_loss_scale
        self.add_patch_pos = add_patch_pos

        self.add_patch_pos = add_patch_pos
        if self.add_patch_pos:
            patch_dim = self.projector.embed_dim
            self.patch_pos_embed = nn.Parameter(
                (patch_dim**-0.5) * torch.randn(4, patch_dim))

    def forward_images(self, images):
        if self.freeze_vision_encoder:
            self.vision_encoder.eval()
            with torch.no_grad():
                image_embeds = self.vision_encoder(images)
        else:
            image_embeds = self.vision_encoder(images)
        return image_embeds

    def forward(self,
                input_ids,
                images,
                attention_mask,
                labels,
                embeds_gen_mask,
                embeds_cmp_mask,
                ids_gen_mask,
                ids_cmp_mask,
                patch_positions=None):

        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        # bz x seq_len x dim, 4 x 160 x 4096
        use_fake_images = False
        bz, sq, dim = input_embeds.shape
        image_embeds = None
        if images is None:
            images = torch.randn(1, 3, 384, 384).to(input_embeds.device,
                                                    dtype=input_embeds.dtype)
            use_fake_images = True
        image_embeds = self.forward_images(images)
        del images

        if not use_fake_images and image_embeds is not None:
            image_embeds_cmp = image_embeds[embeds_cmp_mask]
            # num_imgs_in_batch x nq_in x dim_in, 4 x 64 x 4096
            if patch_positions is not None:
                patch_positions = patch_positions[embeds_cmp_mask]

        if not use_fake_images and image_embeds_cmp.shape[0] > 0:
            image_embeds_lm = self.projector(
                image_embeds_cmp
            )  # num_imgs_in_batch x nq x dim, 4 x 64 x 4096
            if self.add_patch_pos and patch_positions is not None:
                # assert patch_positions is not None
                patch_positions = patch_positions.to(image_embeds_lm)
                rel_pos_embed = torch.mm(
                    torch.cat([patch_positions, 1 - patch_positions], dim=-1) /
                    2, self.patch_pos_embed).unsqueeze(1)
                image_embeds_lm = image_embeds_lm + rel_pos_embed
        else:
            if image_embeds is not None:
                del image_embeds
            image_embeds = None
            image_embeds_cmp_fake = torch.randn(1, 729, 1152).to(
                input_embeds.device, dtype=input_embeds.dtype)
            image_embeds_lm = self.projector(image_embeds_cmp_fake)
            if self.add_patch_pos:
                rel_pos_embed = self.patch_pos_embed.mean(
                    0, keepdim=True).unsqueeze(1)  # 1, 1, dim
                image_embeds_lm = image_embeds_lm + rel_pos_embed

        has_image = image_embeds is not None and embeds_cmp_mask.sum().item(
        ) > 0

        if has_image:
            input_embeds[ids_cmp_mask] = image_embeds_lm.reshape(-1, dim)
            # eg, 128 x 4096
        else:
            input_embeds[:1, :self.projector.
                         num_queries, :] += 0.0 * image_embeds_lm[:1, :, :]
        output_lm = self.language_model(attention_mask=attention_mask,
                                        inputs_embeds=input_embeds,
                                        labels=labels,
                                        output_hidden_states=True,
                                        return_dict=True)
        lm_loss = output_lm['loss']

        total_loss = self.lm_loss_scale * lm_loss
        return {
            'total_loss': total_loss,
            'lm_loss': lm_loss,
        }

    def generate(self,
                 input_ids,
                 pixel_values=None,
                 image_masks=None,
                 image_id_masks=None,
                 attention_mask=None,
                 logits_processor=None,
                 temperature=0.7,
                 num_beams=1,
                 max_new_tokens=120,
                 top_p=0.5,
                 dtype=torch.float16,
                 device='cuda',
                 patch_positions=None,
                 pad_token_id=128001):

        generation_config = {
            'temperature': temperature,
            'num_beams': num_beams,
            'max_new_tokens': max_new_tokens,
            'top_p': top_p,
            'do_sample': False
        }

        input_ids = input_ids.to(device=device)
        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        bz, sq, dim = input_embeds.shape

        if pixel_values is not None:
            assert image_id_masks is not None and image_masks is not None
            with torch.no_grad():
                image_embeds = self.forward_images(pixel_values)
                image_embeds_lm = self.projector(image_embeds)
                if self.add_patch_pos:
                    assert patch_positions is not None
                    patch_positions = patch_positions.to(image_embeds_lm)
                    rel_pos_embed = torch.mm(
                        torch.cat([patch_positions, 1 - patch_positions],
                                  dim=-1) / 2,
                        self.patch_pos_embed).unsqueeze(1)
                    image_embeds_lm = image_embeds_lm + rel_pos_embed

            input_embeds[image_id_masks] = image_embeds_lm[image_masks].view(
                -1, dim)

        output = self.language_model.generate(
            input_ids=input_ids,
            inputs_embeds=input_embeds,
            output_hidden_states=True,
            return_dict_in_generate=True,
            logits_processor=logits_processor,
            attention_mask=attention_mask,
            pad_token_id=pad_token_id,
            **generation_config)
        generate_ids = output.sequences[0][input_ids.shape[1]:]
        return generate_ids

    @classmethod
    def from_pretrained(cls,
                        language_model,
                        vision_encoder,
                        projector,
                        pretrained_model_name_or_path=None,
                        **kwargs):
        model = cls(language_model=language_model,
                    vision_encoder=vision_encoder,
                    projector=projector,
                    **kwargs)
        if os.environ.get('DEBUG_FLAG', 'False') == 'True':
            return model

        if pretrained_model_name_or_path is not None:

            ckpt = torch.load(pretrained_model_name_or_path,
                              map_location='cpu')
            print(ckpt.keys())
            load_zero3_checkpoint(model, ckpt)
        return model


class SEED(GeneraliazedMultimodalModels):
    """_summary_

    Args:
        GeneraliazedMultimodalModels (_type_): _description_
    """

    def __init__(self,
                 language_model,
                 vision_encoder,
                 projector,
                 output_projector,
                 freeze_vision_encoder=True,
                 lm_loss_scale=1.0,
                 rec_loss_scale=1.0,
                 add_patch_pos=False,
                 vit_down=False,
                 mse=False) -> None:
        super().__init__(language_model=language_model,
                         vision_encoder=vision_encoder,
                         projector=projector,
                         freeze_vision_encoder=freeze_vision_encoder,
                         lm_loss_scale=lm_loss_scale,
                         add_patch_pos=add_patch_pos)
        self.output_projector = output_projector
        self.rec_loss_scale = rec_loss_scale
        self.vit_down = vit_down
        if self.vit_down:
            self.pool_size = 4
            self.stride = 4
        self.mse = mse
        if self.mse:
            self.mse_loss = torch.nn.MSELoss()

    def forward(self,
                input_ids,
                images,
                attention_mask,
                labels,
                embeds_gen_mask,
                embeds_cmp_mask,
                ids_gen_mask,
                ids_cmp_mask,
                patch_positions=None):

        input_embeds = self.language_model.get_input_embeddings()(
            input_ids)  # bz x seq_len x dim, 4 x 160 x 4096

        bz, sq, dim = input_embeds.shape

        use_fake_images = False
        bz, sq, dim = input_embeds.shape
        image_embeds = None
        if images is None:
            images = torch.randn(1, 3, 384, 384).to(input_embeds.device,
                                                    dtype=input_embeds.dtype)
            use_fake_images = True
        image_embeds = self.forward_images(images)
        del images

        if not use_fake_images and image_embeds is not None:
            image_embeds_cmp = image_embeds[
                embeds_cmp_mask]  # num_imgs_in_batch x nq_in x dim_in, 4 x 64 x 4096
            if patch_positions is not None:
                patch_positions = patch_positions[embeds_cmp_mask]

        if not use_fake_images and image_embeds is not None and image_embeds_cmp.shape[
                0] > 0:
            image_embeds_lm = self.projector(
                image_embeds_cmp
            )  # num_imgs_in_batch x nq x dim, 4 x 64 x 4096
            if self.add_patch_pos and patch_positions is not None:
                # assert patch_positions is not None
                patch_positions = patch_positions.to(image_embeds_lm)
                rel_pos_embed = torch.mm(
                    torch.cat([patch_positions, 1 - patch_positions], dim=-1) /
                    2, self.patch_pos_embed).unsqueeze(1)
                image_embeds_lm = image_embeds_lm + rel_pos_embed
            has_image_cmp = True
        else:
            image_embeds_cmp_fake = torch.randn(
                1, self.output_projector.num_queries,
                self.output_projector.embed_dim).to(input_embeds.device,
                                                    dtype=input_embeds.dtype)

            image_embeds_lm = self.projector(image_embeds_cmp_fake)
            if self.add_patch_pos:
                rel_pos_embed = self.patch_pos_embed.mean(
                    0, keepdim=True).unsqueeze(1)  # 1, 1, dim
                image_embeds_lm = image_embeds_lm + rel_pos_embed

            has_image_cmp = False

        has_image_input = image_embeds is not None and embeds_cmp_mask.sum(
        ).item() > 0
        has_image_output = image_embeds is not None and embeds_gen_mask.sum(
        ).item() > 0

        if has_image_input:
            input_embeds[ids_cmp_mask] = image_embeds_lm.reshape(
                -1, dim)  # eg, 128 x 4096
        else:
            input_embeds[:1, :self.projector.
                         num_queries, :] += 0.0 * image_embeds_lm[:1, :, :]

        output_lm = self.language_model(attention_mask=attention_mask,
                                        inputs_embeds=input_embeds,
                                        labels=labels,
                                        output_hidden_states=True,
                                        return_dict=True)
        lm_loss = output_lm['loss']

        last_hidden_state = output_lm.hidden_states[-1]  # 4 x 160 x 4096

        if has_image_output:
            target_embeds = image_embeds[
                embeds_gen_mask]  # num_imgs_gen_target x nq_in x dim_in, 2 x 256 x 4096

            if self.vit_down:
                target_embeds = target_embeds.permute(0, 2, 1)  # NLD -> NDL
                target_embeds = F.avg_pool1d(target_embeds,
                                             kernel_size=self.pool_size,
                                             stride=self.stride)
                target_embeds = target_embeds.permute(0, 2, 1)

            num_imgs_for_rec = target_embeds.shape[0]
            output_image_embeds = last_hidden_state[ids_gen_mask].view(
                num_imgs_for_rec, -1, dim)  # 128 x 4096 -> 2 x 64 x 4096

            recon_image_embeds = self.output_projector(
                output_image_embeds)  # 2 x 256 x 4096

            if self.mse:
                rec_loss = F.mse_loss(
                    recon_image_embeds,
                    target_embeds.detach())  # for zero3 compatibility
            else:
                rec_loss = cosine_loss(recon_image_embeds,
                                       target_embeds.detach())

        else:
            output_image_embeds = torch.randn(
                1, self.projector.num_queries, self.projector.embed_dim).to(
                    input_embeds.device, dtype=input_embeds.dtype
                ) + 0.0 * last_hidden_state[0, :self.projector.num_queries, :]
            recon_image_embeds = self.output_projector(output_image_embeds)
            rec_loss = 0.0 * recon_image_embeds.sum()

        total_loss = self.lm_loss_scale * lm_loss + self.rec_loss_scale * rec_loss

        return {
            'total_loss': total_loss,
            'lm_loss': lm_loss,
            'rec_loss': rec_loss
        }

    def generate(self,
                 input_ids,
                 pixel_values=None,
                 embeds_cmp_mask=None,
                 ids_cmp_mask=None,
                 logits_processor=None,
                 num_img_gen_tokens=64,
                 temperature=0.7,
                 num_beams=1,
                 max_new_tokens=120,
                 top_p=0.5,
                 dtype=torch.float16,
                 device='cuda',
                 tokenizer=None,
                 patch_positions=None):
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
            logits_processor.append(
                AutoImageTokenGenerationProcessor(
                    tokenizer=tokenizer,
                    num_img_gen_tokens=num_img_gen_tokens))

        # if prompt is not None:
        #     input_ids = tokenizer(prompt, return_tensors="pt").input_ids

        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids)

        input_ids = input_ids.to(device=device)
        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        bz, sq, dim = input_embeds.shape

        if pixel_values is not None:
            assert embeds_cmp_mask is not None and ids_cmp_mask is not None
            with torch.no_grad():
                image_embeds = self.forward_images(pixel_values)
                image_embeds_lm = self.projector(image_embeds)
                if self.add_patch_pos:
                    assert patch_positions is not None
                    patch_positions = patch_positions.to(image_embeds_lm)
                    rel_pos_embed = torch.mm(
                        torch.cat([patch_positions, 1 - patch_positions],
                                  dim=-1) / 2,
                        self.patch_pos_embed).unsqueeze(1)
                    image_embeds_lm = image_embeds_lm + rel_pos_embed
            input_embeds[ids_cmp_mask] = image_embeds_lm[embeds_cmp_mask].view(
                -1, dim)

        generation_config = {
            'temperature': temperature,
            'num_beams': num_beams,
            'max_new_tokens': max_new_tokens,
            'top_p': top_p,
            'do_sample': False
        }

        output = self.language_model.generate(
            input_ids=input_ids,
            inputs_embeds=input_embeds,
            output_hidden_states=True,
            return_dict_in_generate=True,
            logits_processor=logits_processor,
            **generation_config)

        generate_ids = output.sequences[0][input_ids.shape[1]:]
        generate_id_list = generate_ids.tolist()
        boi_token_id = tokenizer.encode(BOI_TOKEN, add_special_tokens=False)[0]
        eoi_token_id = tokenizer.encode(EOI_TOKEN, add_special_tokens=False)[0]

        last_hidden_states = torch.cat(
            [hidden_state[-1] for hidden_state in output.hidden_states],
            dim=1)[0, input_ids.shape[1]:, :]

        eoi_indices = torch.where(generate_ids == eoi_token_id)[0].tolist()
        num_gen_imgs = len(eoi_indices)
        text_mask = torch.ones_like(generate_ids, dtype=torch.bool)
        has_img_output = num_gen_imgs > 0
        if has_img_output:
            img_gen_feats = []
            for eoi_idx in eoi_indices:
                img_gen_feats.append(
                    last_hidden_states[eoi_idx - num_img_gen_tokens:eoi_idx])
                text_mask[eoi_idx - num_img_gen_tokens:eoi_idx] = False

            img_gen_feats = torch.stack(img_gen_feats)
            img_gen_feat = self.output_projector(img_gen_feats)
        else:
            img_gen_feat = None

        text_mask[generate_ids == boi_token_id] = False
        generate_ids = generate_ids[text_mask]
        generate_text = tokenizer.decode(generate_ids,
                                         skip_special_tokens=False)

        return {
            'text': generate_text,
            'has_img_output': has_img_output,
            'img_gen_feat': img_gen_feat,
            'num_gen_imgs': num_gen_imgs
        }

    @classmethod
    def from_pretrained(cls,
                        language_model,
                        vision_encoder,
                        projector,
                        output_projector,
                        pretrained_model_name_or_path=None,
                        **kwargs):
        model = cls(language_model=language_model,
                    vision_encoder=vision_encoder,
                    projector=projector,
                    output_projector=output_projector,
                    **kwargs)

        if os.environ.get('DEBUG_FLAG', 'False') == 'True':
            return model

        if pretrained_model_name_or_path is not None:
            ckpt = torch.load(pretrained_model_name_or_path, map_location='cpu')
            load_zero3_checkpoint(model, ckpt)

        return model

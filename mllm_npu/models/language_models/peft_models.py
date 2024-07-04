import os

import hydra
import torch
import deepspeed
from omegaconf import DictConfig
from peft import PeftModel, get_peft_model
from transformers import LlamaForCausalLM
try:
    from transformers import MistralForCausalLM as Mist
except Exception as e:
    Mist = None


def get_peft_model_with_resize_embedding(model,
                                         peft_config=None,
                                         model_id=None,
                                         vocab_size=None,
                                         torch_dtype='bf16'):
    if torch_dtype == 'bf16' or torch_dtype == 'bfloat16':
        torch_dtype = torch.bfloat16
    elif torch_dtype == 'fp16' or torch_dtype == 'float16':
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    if isinstance(model, DictConfig):
        if os.environ.get('DEBUG_FLAG', 'False') == 'True':
            if 'mistral' in model['pretrained_model_name_or_path'].lower():
                from transformers import MistralConfig
                config = MistralConfig.from_pretrained(
                    os.path.join(model['pretrained_model_name_or_path'],
                                 'config_debug.json'))
                assert Mist is not None
                model = Mist(config)
            else:
                from transformers import LlamaConfig
                config = LlamaConfig.from_pretrained(
                    os.path.join(model['pretrained_model_name_or_path'],
                                 'config_debug.json'))
                model = LlamaForCausalLM(config)

        else:
            model = hydra.utils.instantiate(model, torch_dtype=torch_dtype)

    assert (peft_config is None) + (model_id is None) == 1

    if vocab_size is not None:
        old_vocab_size = model.config.vocab_size
        if old_vocab_size != vocab_size:
            print(
                f'Old vocab size: {old_vocab_size}, new vocab size: {vocab_size}'
            )

        print(f'Length of tokenizer and resize embedding: {vocab_size}')
        model.resize_token_embeddings(vocab_size)

        if vocab_size - old_vocab_size > 0:
            with deepspeed.zero.GatheredParameters(
                    list(model.get_input_embeddings().parameters()) +
                    list(model.get_output_embeddings().parameters()),
                    modifier_rank=0):

                input_embeddings = model.get_input_embeddings().weight.data
                input_embeddings_avg = input_embeddings[:-vocab_size +
                                                        old_vocab_size].mean(
                                                            dim=0,
                                                            keepdim=True)

                input_embeddings[-vocab_size +
                                 old_vocab_size:] = input_embeddings_avg

                if model.get_output_embeddings(
                ) is not None and not model.config.tie_word_embeddings:
                    output_embeddings = model.get_output_embeddings(
                    ).weight.data
                    output_embeddings_avg = output_embeddings[:-vocab_size +
                                                              old_vocab_size].mean(
                                                                  dim=0,
                                                                  keepdim=True
                                                              ) * 3
                    output_embeddings[-vocab_size +
                                      old_vocab_size:] = output_embeddings_avg

    if peft_config is not None:
        print('peft config: ', peft_config)
        if isinstance(peft_config, DictConfig):
            peft_config = hydra.utils.instantiate(peft_config)
        peft_model = get_peft_model(model=model, peft_config=peft_config)
        peft_model.get_input_embeddings().requires_grad_(True)
        peft_model.get_output_embeddings().requires_grad_(True)

        peft_model.print_trainable_parameters()

    else:
        peft_model = PeftModel.from_pretrained(model=model, model_id=model_id)

    return peft_model


def get_model_with_resize_embedding(model,
                                    vocab_size=None,
                                    torch_dtype='bf16'):
    if torch_dtype == 'bf16' or torch_dtype == 'bfloat16':
        torch_dtype = torch.bfloat16
    elif torch_dtype == 'fp16' or torch_dtype == 'float16':
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    if isinstance(model, DictConfig):
        model = hydra.utils.instantiate(model, torch_dtype=torch_dtype)

    model.requires_grad_(False)
    if vocab_size is not None:
        print(f'Length of tokenizer and resize embedding: {vocab_size}')
        model.resize_token_embeddings(vocab_size)
        model.get_input_embeddings().requires_grad_(True)
        model.get_output_embeddings().requires_grad_(True)

    return model


def get_full_model_with_resize_embedding(model,
                                         vocab_size=None,
                                         torch_dtype='bf16'):
    if torch_dtype == 'bf16' or torch_dtype == 'bfloat16':
        torch_dtype = torch.bfloat16
    elif torch_dtype == 'fp16' or torch_dtype == 'float16':
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    if isinstance(model, DictConfig):
        model = hydra.utils.instantiate(model, torch_dtype=torch_dtype)

    if vocab_size is not None:
        print(f'Length of tokenizer and resize embedding: {vocab_size}')
        model.resize_token_embeddings(vocab_size)

    return model

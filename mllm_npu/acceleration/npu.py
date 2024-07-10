import torch
import torch_npu
import numpy as np
import time
from torch_npu.contrib import transfer_to_npu


def test_flash_attn():
    # flash_attn_func
    query = torch.load("query.pt")
    key = torch.load("key.pt")
    value = torch.load("value.pt")

    # softmax_scale = 1.0 / query.shape[2] ** 0.5
    # out = torch_npu.npu_fusion_attention(query.npu(), key.npu(), value.npu(), query.shape[2],
    #                                             input_layout="BSND",
    #                                             keep_prob=1.,
    #                                             scale=softmax_scale)[0]

    # out = torch_npu.npu_fusion_attention(query.npu(), key.npu(), value.npu(), query.shape[2],
    #                                             input_layout="BSND",
    #                                             keep_prob=1.,
    #                                             scale=1/8)[0]

    # softmax_scale = 1.0 / query.shape[2] ** 0.5
    # out = torch_npu.npu_fusion_attention(query.npu(), key.npu(), value.npu(), query.shape[2],
    #                                             input_layout="BSND",
    #                                             keep_prob=1.,
    #                                             scale=softmax_scale,
    #                                             pre_tockens=256,
    #                                             next_tockens=-1)[0]

    softmax_scale = 1.0 / query.shape[2] ** 0.5
    atten_mask = torch.from_numpy(np.triu(np.ones([2048, 2048]), k=1))
    atten_mask = torch.tensor(atten_mask).to(torch.float16).bool().npu()
    out = torch_npu.npu_fusion_attention(query.npu(), key.npu(), value.npu(), query.shape[2],
                                         input_layout="BSND",
                                         scale=softmax_scale,
                                         atten_mask=atten_mask,
                                         sparse_mode=3)[0]

    # flash_attn_varlen_func
    # s = 4

    # query = torch.load("query.pt")
    # key = torch.load("key.pt")
    # value = torch.load("value.pt")
    # cu_seqlens_q = torch.load("cu_seqlens_q.pt")
    # cu_seqlens_k = torch.load("cu_seqlens_k.pt")

    # max_seqlen_q = s
    # max_seqlen_k = s

    # softmax_scale = 1.0 / query.shape[-1] ** 0.5
    # out = torch_npu.npu_fusion_attention(query.npu(), key.npu(), value.npu(), query.shape[1],
    #                                             input_layout="TND",
    #                                             keep_prob=1.,
    #                                             scale=softmax_scale,
    #                                             actual_seq_qlen=cu_seqlens_q[:-1].cpu().tolist(),
    #                                             actual_seq_kvlen=cu_seqlens_k[:-1].cpu().tolist(),
    #                                             sparse_mode=0)[0]

    # atten_mask = torch.from_numpy(np.triu(np.ones([max_seqlen_q, max_seqlen_k]), k=1))
    # atten_mask = torch.tensor(atten_mask).to(torch.float16).bool().npu()
    # out = torch_npu.npu_fusion_attention(query.npu(), key.npu(), value.npu(), query.shape[1],
    #                                             atten_mask=atten_mask,
    #                                             input_layout="TND",
    #                                             keep_prob=1.,
    #                                             scale=softmax_scale,
    #                                             actual_seq_qlen=cu_seqlens_q[:-1].cpu().numpy().tolist(),
    #                                             actual_seq_kvlen=cu_seqlens_k[:-1].cpu().numpy().tolist(),
    #                                             pre_tockens=2147483647,
    #                                             next_tockens=0)[0]

    print(out)


def test_xformers():
    q = torch.load("q.pt")
    k = torch.load("k.pt")
    v = torch.load("v.pt")

    B, M, H, K = 3, 32, 8, 128

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    q = q.npu()
    k = k.npu()
    v = v.npu()

    out_gpa = torch_npu.npu_fusion_attention(
        q, k, v, 8, input_layout="BNSD",
        pse=None,
        scale=1.0 / q.shape[-1] ** 0.5,
        pre_tockens=2147483647,
        next_tockens=2147483647,
        keep_prob=1.,
        sync=False,
        inner_precise=0,
    )[0]

    print(out_gpa)
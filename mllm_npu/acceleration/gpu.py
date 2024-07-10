
def test_flash_attn():
    import torch
    import numpy as np
    import time
    from flash_attn import flash_attn_func, flash_attn_varlen_func

    # flash_attn_func
    query = torch.randn(4, 8, 128, 128, dtype=torch.float16)
    key = torch.randn(4, 8, 128, 128, dtype=torch.float16)
    value = torch.randn(4, 8, 128, 128, dtype=torch.float16)

    torch.save(query, "query.pt")
    torch.save(key, "key.pt")
    torch.save(value, "value.pt")

    # out = flash_attn_func(query.to("cuda"), key.to("cuda"), value.to("cuda"), dropout_p=0.0)
    # out = flash_attn_func(query.to("cuda"), key.to("cuda"), value.to("cuda"), dropout_p=0.0, softmax_scale=1/8)
    # out = flash_attn_func(query.to("cuda"), key.to("cuda"), value.to("cuda"), dropout_p=0.0, window_size=(256, -1))
    out = flash_attn_func(query.to("cuda"), key.to("cuda"), value.to("cuda"), causal=True)

    # flash_attn_varlen_func
    # b = 2
    # s = 4
    # n = 6
    # d = 128
    # query = torch.randn(b*s, n, d, dtype=torch.float16)
    # key = torch.randn(b*s, n, d, dtype=torch.float16)
    # value = torch.randn(b*s, n, d, dtype=torch.float16)

    # cu_seqlens_q = torch.arange(s, (b+2)*s, s, dtype=torch.int32)
    # cu_seqlens_k = torch.arange(s, (b+2)*s, s, dtype=torch.int32)

    # max_seqlen_q = s
    # max_seqlen_k = s

    # torch.save(query, "query.pt")
    # torch.save(key, "key.pt")
    # torch.save(value, "value.pt")
    # torch.save(cu_seqlens_q, "cu_seqlens_q.pt")
    # torch.save(cu_seqlens_k, "cu_seqlens_k.pt")

    # out = flash_attn_varlen_func(query.to("cuda"), key.to("cuda"), value.to("cuda"),
    #                                 dropout_p=0.0,
    #                                 cu_seqlens_q=cu_seqlens_q.to("cuda"),
    #                                 cu_seqlens_k=cu_seqlens_k.to("cuda"),
    #                                 max_seqlen_q=max_seqlen_q,
    #                                 max_seqlen_k=max_seqlen_k)

    # out = flash_attn_varlen_func(query.to("cuda"), key.to("cuda"), value.to("cuda"),
    #                                 dropout_p=0.0,
    #                                 cu_seqlens_q=cu_seqlens_q.to("cuda"),
    #                                 cu_seqlens_k=cu_seqlens_k.to("cuda"),
    #                                 max_seqlen_q=max_seqlen_q,
    #                                 max_seqlen_k=max_seqlen_k,
    #                                 causal=True)

    print(out)


def test_xformers():
    import xformers.ops as xops

    B, M, H, K = 3, 32, 8, 128

    q = torch.randn([B, M, H, K], dtype=torch.float16)
    k = torch.randn([B, M, H, K], dtype=torch.float16)
    v = torch.randn([B, M, H, K], dtype=torch.float16)

    torch.save(q , "q.pt")
    torch.save(k , "k.pt")
    torch.save(v , "v.pt")

    q = q.to("cuda")
    k = k.to("cuda")
    v = v.to("cuda")

    out_gqa = xops.memory_efficient_attention(q, k, v)

    print(out_gqa)

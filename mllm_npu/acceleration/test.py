import argparse


def cal_time_cpu():
    import torch
    import numpy as np
    import time

    t_cpu = []
    for _ in range(100):
        query = torch.randn(32, 8, 256, 256)
        key = torch.randn(32, 8, 256, 256)
        value = torch.randn(32, 8, 256, 256)

        t1 = time.time()
        out = torch.nn.functional.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
        t2 = time.time()

        t_cpu.append(t2 - t1)

    print("avg time:", sum(t_cpu[20:]) / 80)


def cal_time_npu():
    import torch
    import torch_npu
    import numpy as np
    import time
    from torch_npu.contrib import transfer_to_npu

    t_npu = []
    for _ in range(100):
        query = torch.randn(32, 8, 256, 256, dtype=torch.float16)
        key = torch.randn(32, 8, 256, 256, dtype=torch.float16)
        value = torch.randn(32, 8, 256, 256, dtype=torch.float16)

        query = query.npu()
        key = key.npu()
        value = value.npu()

        softmax_scale = 1.0 / query.shape[2] ** 0.5
        torch_npu.npu.synchronize()
        t1 = time.time()
        out = torch_npu.npu_fusion_attention(query, key, value, query.shape[2],
                                             input_layout="BSND",
                                             keep_prob=1.,
                                             scale=softmax_scale)[0]
        torch_npu.npu.synchronize()
        t2 = time.time()
        t_npu.append(t2 - t1)

    print("avg time:", sum(t_npu[20:]) / 80)


def cal_time_gpu():
    import torch
    import numpy as np
    import time
    from flash_attn import flash_attn_func

    t_gpu = []
    for _ in range(100):
        query = torch.randn(32, 8, 256, 256, dtype=torch.float16)
        key = torch.randn(32, 8, 256, 256, dtype=torch.float16)
        value = torch.randn(32, 8, 256, 256, dtype=torch.float16)

        query = query.to("cuda")
        key = key.to("cuda")
        value = value.to("cuda")

        torch.cuda.synchronize()
        t1 = time.time()
        out = flash_attn_func(query, key, value, dropout_p=0.0)
        torch.cuda.synchronize()
        t2 = time.time()

        t_gpu.append(t2 - t1)

    print("avg time:", sum(t_gpu[20:]) / 80)


def cal_time_gpu_xformers():
    import torch
    import numpy as np
    import time
    import xformers.ops as xops

    t_gpu = []
    for _ in range(100):
        query = torch.randn([32, 256, 8, 256], dtype=torch.float16)
        key = torch.randn([32, 256, 8, 256], dtype=torch.float16)
        value = torch.randn([32, 256, 8, 256], dtype=torch.float16)

        query = query.to("cuda")
        key = key.to("cuda")
        value = value.to("cuda")

        torch.cuda.synchronize()
        t1 = time.time()
        out_gqa = xops.memory_efficient_attention(query, key, value)
        torch.cuda.synchronize()
        t2 = time.time()

        t_gpu.append(t2 - t1)

    print("avg time:", sum(t_gpu[20:]) / 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='attn4npu')

    parser.add_argument("--npu", type=bool, default=False)
    parser.add_argument("--cpu", type=bool, default=False)
    parser.add_argument("--gpu", type=bool, default=False)
    parser.add_argument("--xformers", type=bool, default=False)

    args = parser.parse_args()

    if args.cpu:
        cal_time_cpu()

    if args.npu:
        cal_time_npu()

    if args.gpu:
        if not args.xformers:
            cal_time_gpu()
        else:
            cal_time_gpu_xformers()
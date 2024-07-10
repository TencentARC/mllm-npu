# Acceleration

### Computational cost comparison

This scripts are mainly used to use flash-attn and xformers on NPUs and test them.

| device | function | avg. time cost |
|-------|-------|-------|
| cpu | torch.nn.functional.scaled_dot_product_attention | 0.014441967010498047 |
| npu(910b) | torch_npu.npu_fusion_attention | 0.0022245049476623535 |
| gpu(a100) | flash_attn.flash_attn_func | 0.0007785201072692871 |
| gpu(a100) | xformers.ops.memory_efficient_attention | 0.00046073496341705324 |

### Environment

NPU:
```shell
torch               2.1.0+cpu
torch-npu           2.1.0.post3
```

GPU (flash-attn):
```shell
torch               2.1.0+cu118
flash-attn          2.5.9.post1
numpy               1.26.0
```

GPU (xformers):
```shell
torch               2.3.0+cu118
xformers            0.0.26.post1+cu118
```

For other specific parameter comparison relations, please check the corresponding py file demo. The calculation errors are all 5 decimal places.

### Function comparison

Please refer to the code, now supported

- flash_attn.flash_attn_func (causal)

- flash_attn.flash_attn_varlen_func (causal)

- xformers.ops.memory_efficient_attention (causal)


### References

1）https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/ptmoddevg/trainingmigrguide/performance_tuning_0027.html

2）https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/apiref/apilist/ptaoplist_000448.html
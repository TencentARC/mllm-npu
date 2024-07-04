#!/bin/bash
ps aux | grep 'train/train.py' | awk '{print $2}' | xargs kill -9

echo "----------------------------------------------------"
echo environment initialization
echo "----------------------------------------------------"

EXP_NAME="train_mllm_llama3_8b_siglip_vit_8npus"
PROJ_PATH="mllm_npu"
OUTPUT_PATH="mllm_npu/outputs/${EXP_NAME}"
mkdir -p $OUTPUT_PATH

export PYTHONPATH=/path/to/peft/src/:$PYTHONPATH
export PYTHONPATH=${PYTHONPATH}:/path/to/mllm_npu

export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_CHECKS_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_COLLNET_ENABLE=0
export SHARP_COLL_ENABLE_SAT=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_PXN_DISABLE=1
export NCCL_SOCKET_IFNAME=eth1
export UCX_NET_DEVICES=eth1
export NCCL_IB_HCA=mlx5_eth_1,mlx5_eth_5,mlx5_eth_3,mlx5_eth_7,mlx5_eth_4,mlx5_eth_8,mlx5_eth_2,mlx5_eth_6
export GLOO_SOCKET_IFNAME=eth1

source /usr/local/Ascend/ascend-toolkit/set_env.sh
which python
torchrun --nnodes=1 --nproc_per_node=8 \
    ${PROJ_PATH}/mllm_npu/train/train.py \
    --model ${PROJ_PATH}/mllm_npu/configs/models/mllm_llama3_8b_siglip_vit.yaml \
    --train_dataset ${PROJ_PATH}/mllm_npu/configs/dataset/pretrain_data.yaml \
    --deepspeed_plugin ${PROJ_PATH}/mllm_npu/configs/deepspeed/zero3.json \
    --output_dir ${OUTPUT_PATH} \
    --expr_name ${EXP_NAME} \
    --learning_rate 1e-4 \
    --batch_size 1 \
    --weight_decay 0.05 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-6 \
    --gradient_accumulation_steps 2 \
    --mixed_precision bf16 \
    --num_train_epochs 10 \
    --max_steps 100000 \
    --save_steps 1000 \
    --lr_scheduler_type cosine \
    --warmup_steps 500 \
    --min_lr_ratio 0.05 \
    --dataloader_num_workers 4
    
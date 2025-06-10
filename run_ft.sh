#!/bin/bash

# Check if arguments were provided
if [ $# -lt 2 ]; then
  echo "Usage: $0 <mode> <model_key> [optional_args]"
  echo "  mode: 'fsdp' or 'deepspeed'"
  echo "  model_key: 'qwen', 'llama2-7b', 'llama3.1-8b', 'llama3.1-70b', 'olmo'"
  exit 1
fi

MODE=$1
MODEL_KEY=$2
shift 2  # Remove the first two arguments

# Set environment variables for proper GPU usage
export CUDA_VISIBLE_DEVICES=5,7

if [ "$MODE" = "fsdp" ]; then
  # Create a config file for FSDP
  cat > accelerate_config.yaml << EOF
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
downcast_bf16: 'no'
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch_policy: BACKWARD_PRE
  fsdp_offload_params: false
  fsdp_sharding_strategy: 1
  fsdp_state_dict_type: FULL_STATE_DICT
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
use_cpu: false
EOF

  # Run with accelerate using FSDP config
  accelerate launch --config_file accelerate_config.yaml llm_ft.py \
      --model_key "$MODEL_KEY" \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 4 \
      --num_train_epochs 3 \
      --learning_rate 2e-4 \
      --lr_scheduler_type "cosine" \
      --warmup_ratio 0.03 \
      --weight_decay 0.01 \
      --max_seq_length 128 \
      --lora_r 16 \
      --lora_alpha 32 \
      --lora_dropout 0.05 \
      --seed 42 \
      --chain_nums 2 \
      --fp16 true \
      --save_steps 500 \
      --save_total_limit 1 \
      --logging_steps 10 \
      "$@"

elif [ "$MODE" = "deepspeed" ]; then
  # Create a DeepSpeed config file
  cat > ds_config.json << EOF
{
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "initial_scale_power": 16,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 2e-4,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 2e-4,
      "warmup_num_steps": "auto"
    }
  },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "contiguous_gradients": true
  },
  "gradient_accumulation_steps": 4,
  "train_batch_size": 16,
  "train_micro_batch_size_per_gpu": 4
}
EOF

  # Run with DeepSpeed
  deepspeed llm_ft.py \
      --deepspeed ds_config.json \
      --model_key "$MODEL_KEY" \
      --per_device_train_batch_size 4 \
      --gradient_accumulation_steps 4 \
      --num_train_epochs 3 \
      --learning_rate 2e-4 \
      --lr_scheduler_type "cosine" \
      --warmup_ratio 0.03 \
      --weight_decay 0.01 \
      --max_seq_length 128 \
      --lora_r 16 \
      --lora_alpha 32 \
      --lora_dropout 0.05 \
      --seed 42 \
      --chain_nums 2 \
      --fp16 true \
      --save_steps 500 \
      --save_total_limit 1 \
      --logging_steps 10 \
      "$@"

else
  echo "Invalid mode: $MODE. Use 'fsdp' or 'deepspeed'"
  exit 1
fi

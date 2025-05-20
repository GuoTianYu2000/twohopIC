#!/bin/bash

# Check if an argument was provided
if [ $# -lt 1 ]; then
  echo "Usage: $0 <mode> [optional_args]"
  echo "  mode: 'fsdp' or 'deepspeed'"
  exit 1
fi

MODE=$1
shift  # Remove the first argument

# Define all the models we want to train
MODELS=("qwen" "llama2-7b" "llama3-8b" "olmo" "llama3-70b")

# You can add llama3-70b if your hardware can handle it
# MODELS=("qwen" "llama2-7b" "llama3-8b" "llama3-70b" "olmo")

echo "Starting training for all models using $MODE mode..."

for MODEL in "${MODELS[@]}"; do
  echo "================================================================"
  echo "Training model: $MODEL"
  echo "================================================================"
  
  # Run the training script for the current model
  bash run_ft.sh "$MODE" "$MODEL" "$@"
  
  # Check if the training was successful
  if [ $? -ne 0 ]; then
    echo "Training failed for model: $MODEL"
    echo "Skipping to next model..."
  else
    echo "Training completed successfully for model: $MODEL"
  fi
  
  echo ""
done

echo "All model training completed!" 

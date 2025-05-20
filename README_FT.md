# Language Model Fine-Tuning with HuggingFace Trainer

This repository contains scripts for fine-tuning language models on question-answering tasks using the HuggingFace Trainer API with distributed training on multiple GPUs.

## Setup

1. Make sure you have all the required dependencies installed:

```bash
pip install transformers datasets accelerate peft tqdm
```

2. Make the shell script executable:

```bash
chmod +x run_ft.sh
```

## Data Format

The training data should be in JSON format. The default data file is `qwen2.5/test_short.json`, which has a structure like:

```json
{
  "1": [
    {
      "question": "What is the capital of France?",
      "answer": " Paris",
      "query_names": [1234, 5678, 9012],
      "non_query_names": []
    },
    ...
  ]
}
```

The script processes this data format and constructs instruction-tuning examples where:
- The instruction is "Please answer the question with the most appropriate response."
- The input is the question from the data
- The output is the answer from the data

## Fine-Tuning

### Configuration

Before running the fine-tuning, you may want to customize several parameters in the `run_ft.sh` script:

- `MODEL_NAME`: The model to fine-tune (default: "Qwen/Qwen2.5-7B")
- `OUTPUT_DIR`: Where to save the fine-tuned model (default: "./fine_tuned_model")
- Fine-tuning hyperparameters like learning rate, batch size, etc.

### Running Distributed Training

To run the fine-tuning on 4 GPUs:

```bash
./run_ft.sh
```

This uses the HuggingFace Accelerate library to manage distributed training across 4 GPUs. The configuration is defined in `accelerate_config.yaml`.

### Low-Memory Training with LoRA

By default, the script uses Parameter-Efficient Fine-Tuning (PEFT) with LoRA to reduce memory requirements. This allows fine-tuning large models on limited hardware.

LoRA parameters can be adjusted in the `run_ft.sh` script:
- `--lora_r`: Rank of the LoRA update matrices (default: 16)
- `--lora_alpha`: LoRA alpha parameter (default: 32)
- `--lora_dropout`: Dropout probability for LoRA layers (default: 0.05)

### 4-bit Quantization

The script also supports 4-bit quantization to further reduce memory requirements. This is enabled by default with the `--use_4bit True` flag in the `run_ft.sh` script.

## Evaluation

After fine-tuning, you can evaluate the model using the `evaluate_model.py` script:

```bash
python evaluate_model.py --model_path ./fine_tuned_model --test_file qwen2.5/test_short.json
```

For LoRA models, use:

```bash
python evaluate_model.py --base_model Qwen/Qwen2.5-7B --peft_model ./fine_tuned_model --test_file qwen2.5/test_short.json
```

The evaluation script will:
1. Load the fine-tuned model (either full model or base model + LoRA adapter)
2. Process the test data
3. Generate predictions for each question
4. Compare the predictions with the true answers
5. Calculate and report the accuracy
6. Save detailed results to a JSON file (default: `evaluation_results.json`)

## Tips for Distributed Training

1. The script sets `CUDA_VISIBLE_DEVICES` to use all four GPUs (0,1,2,3).
2. The Accelerate configuration file (`accelerate_config.yaml`) controls how the distributed training is set up.
3. Make sure your machine has enough memory to handle the model, even with the memory-saving techniques.
4. Adjust the `per_device_train_batch_size` and `gradient_accumulation_steps` based on your GPU memory.
5. Monitor GPU memory usage during training and adjust parameters if needed.

## Troubleshooting

If you encounter CUDA out-of-memory errors:
1. Reduce batch size
2. Increase gradient accumulation steps
3. Enable 4-bit quantization if not already enabled
4. Reduce `max_seq_length`

If training is too slow:
1. Make sure your GPUs are properly utilized
2. Check for potential bottlenecks in data loading 

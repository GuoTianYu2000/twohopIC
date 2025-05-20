# Replication Code for "How Do LLMs Perform Two-Hop Reasoning in Context?"

## Environment Setup
```sh
uv sync
source twohop/bin/activate
```



## Three-Layer Transformers
### Getting Started
Before running the scripts, please modify the PROJECT_PATH variable in python files to point to your local repository directory.
```sh
chmod +x scripts/train-3-layer-tf.sh
scripts/train-3-layer-tf.sh
chmod +x scripts/analyze-3-layer-tf.sh
scripts/analyze-3-layer-tf.sh
```

## LLM Finetuning and Evaluation

### Finetuning
The repository supports finetuning various LLM models (Qwen, LLaMA2-7B, LLaMA3-8B, OLMO) using either FSDP (Fully Sharded Data Parallel) or DeepSpeed. The finetuning script uses LoRA for efficient parameter-efficient fine-tuning.

To run finetuning:
```sh
./run_ft.sh <mode> <model_key> [optional_args]
```
- `mode`: Choose between 'fsdp' or 'deepspeed'
- `model_key`: Select from 'qwen', 'llama2-7b', 'llama3-8b', 'llama3-70b', 'olmo'

Example:
```sh
./run_ft.sh fsdp llama2-7b
```


### Evaluation
The evaluation pipeline is implemented in `llm_eval.py` and supports:
- Evaluating both base and LoRA-finetuned models
- Processing multiple chain lengths (up to 5-hop reasoning)
- Topic-based analysis (geography, relations, biology, arithmetic)
- Detailed probability tracking for specific tokens

To run evaluation:
```sh
python llm_eval.py --model_name <model_name> [--use_lora]
```

The evaluation results are saved in model-specific directories with detailed probability distributions and analysis metrics.

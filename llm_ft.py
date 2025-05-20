"""
Simplified LLM Fine-tuning Script with LoRA
"""

import os
import json
import logging
import sys
from dataclasses import dataclass, field
from typing import Optional
import pdb
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)
from accelerate import Accelerator

# Import model configurations
sys.path.append("src")
from llm_eval_configs import MODEL_OPTIONS

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """Arguments pertaining to model configuration."""
    model_key: str = field(
        default="",
        metadata={"help": "Key for the model in MODEL_OPTIONS"}
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA attention dimension"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha parameter"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout probability"}
    )

@dataclass
class DataArguments:
    """Arguments pertaining to data processing."""
    data_dir: str = field(
        default="",
        metadata={"help": "Directory containing the training data"}
    )
    chain_nums: int = field(
        default=2,
        metadata={"help": "The chain_nums value to filter on (1, 2, 3, or 4)"}
    )
    max_seq_length: int = field(
        default=64,
        metadata={"help": "Maximum sequence length for the model"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "Number of processes for preprocessing"},
    )


def load_data(data_dir, chain_nums=None):
    """Load data from a JSON file and filter by chain_nums if provided."""
    file_path = os.path.join(data_dir, "test_long.json")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
        
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if chain_nums is not None:
            filtered_data = []
            for chain_num, items in data.items():
                if int(chain_num) == chain_nums:
                    filtered_data.extend(items)
            return filtered_data
        else:
            # If no chain_nums provided, return all data
            all_data = []
            for items in data.values():
                all_data.extend(items)
            return all_data
            
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in {file_path}")


class TwoHopICData(Dataset):
    def __init__(self, data, tokenizer, max_seq_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.tokenizer.padding_side = 'right'
        self.input_ids = []
        self.attention_mask = []
        self.labels = []

        for example in data:
            question = example.get("question", "")
            answer = example.get("answer", "")

            if not question or not answer:
                logger.warning("Skipping example with missing question or answer")
                continue
        
            question_tokens = self.tokenizer(
                question,
                padding=False,
                return_tensors=None
            )

            all_tokens = self.tokenizer(
                question + answer,
                padding=False,
                return_tensors=None
            )
            
            input_ids = all_tokens["input_ids"]
            attention_mask = all_tokens["attention_mask"]
            labels = [-100] * len(question_tokens["input_ids"]) + input_ids[len(question_tokens["input_ids"]):]

            pad_length = self.max_seq_length - len(input_ids)
            if pad_length > 0:
                input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_length
                attention_mask = attention_mask + [0] * pad_length
                labels = labels + [-100] * pad_length
            else:
                raise ValueError(f"Input ids are longer than max_seq_length: {len(input_ids)}")
        
            self.input_ids.append(input_ids)
            self.attention_mask.append(attention_mask)
            self.labels.append(labels)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        } 

def preprocess_data(data, tokenizer, max_seq_length, preprocessing_num_workers=None):
    """Preprocess the data for training."""
    # Process examples
    examples = []
    for item in data:
        question = item.get("question", "")
        answer = item.get("answer", "")
        
        if not question or not answer:
            logger.warning("Skipping example with missing question or answer")
            continue
        
        examples.append({
            "question": question,
            "answer": answer,
        })
    
    tokenized_dataset = TwoHopICData(examples, tokenizer, max_seq_length)
    
    return tokenized_dataset


def main():
    """Main entry point for the script."""
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Set FSDP to use full_shard with auto_wrap
    training_args.fsdp = "full_shard auto_wrap"
    training_args.fsdp_transformer_layer_cls_to_wrap = None  # Auto wrap will detect transformer layers automatically
    
    # Initialize accelerator
    accelerator = Accelerator()
    is_main_process = accelerator.is_local_main_process
    
    # Configure logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process else logging.WARNING,
    )
    
    # Get model config
    if model_args.model_key not in MODEL_OPTIONS:
        raise ValueError(f"Model key {model_args.model_key} not found in MODEL_OPTIONS")
        
    model_config = MODEL_OPTIONS[model_args.model_key]
    model_name = model_config["name"]
    trust_remote_code = model_config.get("trust_remote_code", True)
    
    # Set output directory
    output_dir = os.path.join("finetuned", f"{model_args.model_key}_chain{data_args.chain_nums}")
    training_args.output_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code
    )
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=trust_remote_code,
    )
    
    # Apply LoRA
    # Define target modules based on model architecture
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]  # Default for most models
    
    # Define LoRA config
    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    
    # Get PEFT model
    model = get_peft_model(model, lora_config)
    if is_main_process:
        logger.info("Applied LoRA configuration")
        model.print_trainable_parameters()
    

    # Load and preprocess data
    data_dir = model_config["dirname"]
    data = load_data(data_dir, data_args.chain_nums)
    
    if is_main_process:
        logger.info(f"Loaded {len(data)} examples with chain_nums={data_args.chain_nums}")
    
    # Prepare dataset
    train_dataset = preprocess_data(
        data, 
        tokenizer, 
        data_args.max_seq_length, 
        data_args.preprocessing_num_workers
    )
    

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    # Train the model
    if is_main_process:
        logger.info("Starting model training")

    trainer.train()
    
    # Save the model (main process only)
    if is_main_process:
        logger.info(f"Saving model to {training_args.output_dir}")
        trainer.save_model(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
    
    # Wait for everyone before returning
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings
    main()

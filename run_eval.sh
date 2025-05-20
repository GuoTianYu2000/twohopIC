#!/bin/bash

# Get the model key from command line args (optional)

echo "Starting evaluation of models..."

python llm_eval.py
python llm_eval.py --lora


echo "Evaluation complete." 

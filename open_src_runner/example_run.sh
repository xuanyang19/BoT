#!/bin/bash

# Example script to run the BOT actor refinement pipeline

# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here" # TODO: Replace with your actual API key

# Navigate to runner_pipeline directory
cd runner_pipeline

# Run on MedQA
python main.py \
  --dataset medqa \
  --split test \
  --batch_size 4 \
  --model gpt-4o-2024-11-20 \
  --limit 8 \
  --out_dir ../outputs_medqa_test

echo "Results saved to outputs_medqa_test/"

# Run on Winogrande
python main.py \
  --dataset winogrande \
  --split test \
  --batch_size 4 \
  --model gpt-4o-2024-11-20 \
  --limit 8 \
  --out_dir ../outputs_winogrande_test

echo "Results saved to outputs_winogrande_test/"

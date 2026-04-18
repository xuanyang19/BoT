#!/bin/bash

# Example script to run the BoT-R pipeline

# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here" # TODO: Replace with your actual API key

# Run on MedQA
python -m bot.main \
  --dataset medqa \
  --split test \
  --batch_size 4 \
  --model gpt-4o-2024-11-20 \
  --limit 8 \
  --out_dir outputs/medqa_test

echo "Results saved to outputs/medqa_test/"

# Run on Winogrande
python -m bot.main \
  --dataset winogrande \
  --split test \
  --batch_size 4 \
  --model gpt-4o-2024-11-20 \
  --limit 8 \
  --out_dir outputs/winogrande_test

echo "Results saved to outputs/winogrande_test/"

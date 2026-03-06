# BOT Actor Refinement - Open Source Runner

Open-source version of the BOT actor refinement pipeline for improving LLM responses through multi-round reflection.

## Quick Start

### 1. Install Dependencies

```bash
pip install openai pandas numpy datasets transformers torch scikit-learn scipy
```

### 2. Set API Key

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 3. Run

```bash
cd runner_pipeline
python main.py --dataset medqa --batch_size 8
```

## Usage

### Basic Command

```bash
python main.py --dataset <dataset> --batch_size 8
```

### Common Options

- `--dataset`: Dataset name (medqa, winogrande)
- `--split`: Data split (train/dev/test, default: test)
- `--batch_size`: Batch size (default: 8)
- `--model`: Model name (default: gpt-4o-2024-11-20)
- `--batching`: Strategy (sequential/random/kmeans, default: sequential)
- `--limit`: Max examples to process (optional)
- `--out_dir`: Output directory (default: temp)

### Example

run `example_run.sh`

## Supported Datasets

- **MedQA**: Medical QA (USMLE multiple choice)
- **Winogrande**: Commonsense reasoning

See `dataloader.py` for the custom dataset interface to add your own datasets.

## Output

Results saved to `<out_dir>/<dataset>_<split>_reflect_<batch_size>_<timestamp>/`:
- `ckpt_20pct_*.json`, `ckpt_40pct_*.json`, ..., `ckpt_final.json`: Checkpoints
- `conversations.jsonl`: Full conversation histories
- `reflections.jsonl`: Reflection histories

## Customization

### Add Custom Models

Edit `runner_pipeline/model.py`:

```python
SUPPORTED_MODELS = [
    "gpt-4o",
    "your-model-name",  # Add here
]
```

### Add Custom Dataset

1. Add loader in `dataloader.py` (see the Custom Dataset Interface section)
2. Add evaluator in `evaluate.py`
3. Add prompt and query builder in `actor_refinement.py`
4. Update routers in all three files

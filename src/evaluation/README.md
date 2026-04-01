# DeepSeek LLM Evaluation Pipeline

High-performance async evaluation pipeline for comparing LLM models using DeepSeek API.

## Features

- **Async Processing**: Parallel evaluation with asyncio
- **DeepSeek R1 Judge**: Chain-of-thought reasoning evaluation
- **Local Embeddings**: Cost-free semantic scoring
- **Cost Tracking**: Real-time pricing calculation
- **Rich Dashboard**: Beautiful console output
- **CSV Export**: Detailed results for analysis

## Quick Start

```bash
# Install dependencies
pip install -r requirements-eval.txt

# Set API key
export DEEPSEEK_API_KEY="sk-your-key"

# Run evaluation
python -m src.evaluation.main
```

## Pricing (DeepSeek 2026)

| Model | Input | Output |
|-------|-------|--------|
| deepseek-chat (V3) | $0.27/1M | $1.10/1M |
| deepseek-reasoner (R1) | $0.55/1M | $2.19/1M |

## Project Structure

```
src/evaluation/
├── __init__.py
├── config.py        # Pricing & model configs
├── async_client.py  # Async DeepSeek client
├── dataset.py       # Dataset loader
├── scorer.py        # Local embedding scorer
├── judge.py         # DeepSeek R1 judge
├── cost.py          # Cost tracking
├── dashboard.py     # Rich console output
├── main.py          # Entry point
```

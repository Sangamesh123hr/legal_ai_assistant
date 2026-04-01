# DeepSeek LLM Evaluation Pipeline

High-performance async evaluation system for comparing LLM models using DeepSeek API with chain-of-thought reasoning.

## Features

- **Async Processing** - Parallel evaluation with asyncio + httpx
- **DeepSeek R1 Judge** - Chain-of-thought reasoning evaluation
- **Local Embeddings** - Cost-free semantic scoring (sentence-transformers)
- **Cost Tracking** - Real-time pricing calculation
- **Beautiful Output** - Rich terminal dashboard with tables

## Quick Start

```bash
# Install dependencies
pip install -r requirements-eval.txt

# Run fast evaluation
python run_fast.py

# Run full evaluation (with judge)
python run_eval.py
```

## Pricing (DeepSeek 2026)

| Model | Input | Output |
|-------|-------|--------|
| deepseek-chat (V3) | $0.27/1M | $1.10/1M |
| deepseek-reasoner (R1) | $0.55/1M | $2.19/1M |

## Benchmark Results

| Model | Avg Cosine | Latency | Cost/sample |
|-------|-----------|---------|------------|
| DeepSeek V3 | 0.770 | ~3s | $0.0001 |
| DeepSeek R1 | 0.754 | ~7s | $0.0004 |

## Project Structure

```
src/evaluation/
├── async_client.py  # Async DeepSeek client
├── scorer.py        # Local embedding scorer
├── judge.py         # DeepSeek R1 judge
├── cost.py          # Cost tracking
├── dataset.py       # Dataset loader
├── config.py        # Configuration
└── main.py          # Entry point
```

## Evaluation Metrics

- **Cosine Similarity** - Semantic similarity using local embeddings
- **LLM-as-Judge** - Quality scored by DeepSeek R1
- **Latency** - Response time in milliseconds
- **Cost** - Token-based cost estimation

---

Built with asyncio + DeepSeek API

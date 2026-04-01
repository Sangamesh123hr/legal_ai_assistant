"""
Fast runner for DeepSeek LLM Evaluation Pipeline (No Judge)
"""

import asyncio
import os
import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from src.evaluation.config import EvalConfig, EvalSample
from src.evaluation.async_client import DeepSeekAsyncClient
from src.evaluation.scorer import LocalEmbeddingScorer
from src.evaluation.cost import CostTracker

API_KEY = "sk-d6aaf104a9104ca4a19553b1a304eb9c"

SYSTEM_PROMPT = """You are an expert legal AI assistant. Answer questions based ONLY on the provided context. 
Be precise and cite relevant details. If information is not in the context, say so."""

SAMPLES = [
    EvalSample(
        id="legal_001",
        question="What are the payment terms?",
        context="Total: $150,000. 30% on signing ($45,000). 40% at milestone ($60,000). 30% at delivery ($45,000).",
        ground_truth="Total $150,000. 30% ($45,000) signing, 40% ($60,000) milestone, 30% ($45,000) delivery.",
    ),
    EvalSample(
        id="legal_002",
        question="What is the termination clause?",
        context="Either party may terminate for breach. Client may terminate with 30 days notice. Completed work must be delivered.",
        ground_truth="Termination for breach or 30 days notice. Completed work delivered.",
    ),
    EvalSample(
        id="legal_003",
        question="Who owns the IP?",
        context="IP rights transfer to Client upon final payment. Developer retains general knowledge rights.",
        ground_truth="IP transfers to Client on final payment. Developer keeps general knowledge rights.",
    ),
]


async def main():
    print("\n" + "=" * 50)
    print("  DeepSeek LLM Evaluation - FAST MODE")
    print("=" * 50 + "\n")

    config = EvalConfig(api_key=API_KEY)
    scorer = LocalEmbeddingScorer()
    cost_tracker = CostTracker()

    models = [
        ("DeepSeek V3", "deepseek-chat"),
        ("DeepSeek R1", "deepseek-reasoner"),
    ]

    async with DeepSeekAsyncClient(config) as client:
        for model_name, model_id in models:
            print(f"\n[ {model_name} ]")
            print("-" * 40)

            scores = []

            for sample in SAMPLES:
                prompt = f"Context: {sample.context}\n\nQuestion: {sample.question}\n\nAnswer:"

                print(f"  {sample.id}: ", end="", flush=True)

                response = await client.chat(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    system_prompt=SYSTEM_PROMPT,
                )

                cost_tracker.record(model_name, response)

                if response.error:
                    print(f"ERROR: {response.error}")
                    continue

                # Calculate semantic similarity (FREE!)
                metrics = scorer.calculate_metrics(
                    response.content, sample.ground_truth
                )
                cosine = metrics["cosine_similarity"]

                scores.append(cosine)
                print(
                    f"Cosine: {cosine:.3f} | Latency: {response.latency_ms:.0f}ms | Cost: ${response.cost_usd:.4f}"
                )

                # Show R1 reasoning
                if response.reasoning:
                    print(f"         Reasoning: {response.reasoning[:80]}...")

            if scores:
                print(f"\n  Average Cosine: {sum(scores) / len(scores):.3f}")

    # Summary
    print("\n" + "=" * 50)
    print("  COST SUMMARY")
    print("=" * 50)

    total = cost_tracker.total_cost
    print(f"\n  Total Cost: ${total:.4f}")
    print(
        f"  Total Tokens: {cost_tracker.total_tokens['input']:,} in / {cost_tracker.total_tokens['output']:,} out"
    )
    print("\n  Done!")


if __name__ == "__main__":
    asyncio.run(main())

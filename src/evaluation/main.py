"""
DeepSeek LLM Evaluation Pipeline - Main Entry Point

Run with:
    python -m src.evaluation.main
"""

import asyncio
import csv
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.config import EvalConfig, EvalSample, DEFAULT_MODELS
from src.evaluation.async_client import DeepSeekAsyncClient
from src.evaluation.scorer import LocalEmbeddingScorer
from src.evaluation.judge import DeepSeekJudge
from src.evaluation.cost import CostTracker
from src.evaluation.dashboard import EvaluationDashboard
from src.evaluation.dataset import DatasetLoader


SYSTEM_PROMPT = """You are an expert legal AI assistant. Answer questions based ONLY on the provided context. 
Be precise and cite relevant details. If information is not in the context, say so."""


class EvaluationPipeline:
    """Main evaluation pipeline."""

    def __init__(self, api_key: str):
        self.config = EvalConfig(api_key=api_key)
        self.dashboard = EvaluationDashboard()
        self.cost_tracker = CostTracker()
        self.scorer = LocalEmbeddingScorer()
        self.results: Dict[str, List[Dict]] = {}

    async def run(self, models: List[Dict] = None, num_samples: int = 8):
        """Run the full evaluation pipeline."""

        # Initialize
        self.dashboard.print_header()
        logger.info("Starting DeepSeek LLM Evaluation Pipeline")

        # Load dataset
        dataset = DatasetLoader()
        samples = dataset.load(limit=num_samples)
        logger.info(f"Loaded {len(samples)} evaluation samples")

        # Use default models if none specified
        if models is None:
            models = DEFAULT_MODELS

        # Initialize DeepSeek client
        async with DeepSeekAsyncClient(self.config) as client:
            # Initialize judge
            judge = DeepSeekJudge(client)

            # Evaluate each model
            for model_config in models:
                model_name = model_config["name"]
                model_id = model_config["model_id"]

                logger.info(f"\nEvaluating: {model_name} ({model_id})")

                results = await self._evaluate_model(
                    client, judge, samples, model_name, model_id
                )
                self.results[model_name] = results

        # Generate reports
        self._save_csv()
        self.dashboard.print_final_report(self.results, self.cost_tracker)

        return self.results

    async def _evaluate_model(
        self,
        client: DeepSeekAsyncClient,
        judge: DeepSeekJudge,
        samples: List[EvalSample],
        model_name: str,
        model_id: str,
    ) -> List[Dict]:
        """Evaluate a single model on all samples."""

        results = []
        progress = self.dashboard.create_progress()

        with progress:
            task = progress.add_task(
                f"[cyan]Evaluating {model_name}[/cyan]", total=len(samples)
            )

            for sample in samples:
                try:
                    # Build prompt
                    prompt = self._build_prompt(sample)

                    # Call model
                    response = await client.chat(
                        model=model_id,
                        messages=[{"role": "user", "content": prompt}],
                        system_prompt=SYSTEM_PROMPT,
                    )

                    # Track cost
                    self.cost_tracker.record(model_name, response)

                    if response.error:
                        logger.error(f"Error: {response.error}")
                        results.append(
                            {
                                "sample_id": sample.id,
                                "error": response.error,
                            }
                        )
                        progress.update(task, advance=1)
                        continue

                    # Calculate semantic metrics (free!)
                    metrics = self.scorer.calculate_metrics(
                        response.content, sample.ground_truth
                    )

                    # Judge evaluation (costs API credits)
                    judge_result = await judge.evaluate(
                        question=sample.question,
                        context=sample.context,
                        response=response.content,
                        ground_truth=sample.ground_truth,
                    )

                    result = {
                        "sample_id": sample.id,
                        "question": sample.question,
                        "ground_truth": sample.ground_truth,
                        "response": response.content,
                        "reasoning": response.reasoning,  # R1 chain-of-thought
                        "metrics": metrics,
                        "judge_score": judge_result.score,
                        "judge_reasoning": judge_result.reasoning,
                        "hallucination_risk": judge_result.hallucinations,
                        "latency_ms": response.latency_ms,
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                        "cost_usd": response.cost_usd,
                        "category": sample.category,
                        "difficulty": sample.difficulty,
                    }

                    results.append(result)
                    self.dashboard.print_sample_result(sample.id, model_name, result)

                except Exception as e:
                    logger.error(f"Error on sample {sample.id}: {e}")
                    results.append({"sample_id": sample.id, "error": str(e)})

                progress.update(task, advance=1)

        return results

    def _build_prompt(self, sample: EvalSample) -> str:
        """Build prompt from sample."""
        return f"""Based on the following context, answer the question.

CONTEXT:
{sample.context}

QUESTION:
{sample.question}

ANSWER:"""

    def _save_csv(self):
        """Save results to CSV."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(self.config.results_dir)
        results_dir.mkdir(exist_ok=True)

        csv_path = results_dir / f"eval_results_{timestamp}.csv"

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            if not self.results:
                return

            # Get all keys from first result
            first_result = self.results[list(self.results.keys())[0]][0]
            fieldnames = list(first_result.keys())

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for model_name, results in self.results.items():
                for result in results:
                    result["model"] = model_name
                    writer.writerow(result)

        logger.info(f"Saved results to {csv_path}")


async def main():
    """Main entry point."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")

    if not api_key:
        api_key = input("Enter your DeepSeek API key: ").strip()

    if not api_key:
        print("Error: API key required!")
        sys.exit(1)

    pipeline = EvaluationPipeline(api_key)
    await pipeline.run()


if __name__ == "__main__":
    asyncio.run(main())

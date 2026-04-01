"""
Results Reporter - Visualization and Reporting
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ResultsReporter:
    """Generate reports and visualizations from evaluation results."""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def generate_report(
        self,
        all_results: Dict[str, List[Dict]],
        aggregated: Dict[str, Dict],
        config: Any,
    ) -> Dict[str, str]:
        """
        Generate all reports from evaluation results.

        Returns:
            Dictionary with paths to generated reports
        """
        reports = {}

        # 1. Save raw results to CSV
        csv_path = self._save_csv(all_results)
        reports["csv"] = str(csv_path)

        # 2. Generate leaderboard
        leaderboard_path = self._generate_leaderboard(aggregated)
        reports["leaderboard"] = str(leaderboard_path)

        # 3. Generate HTML report
        html_path = self._generate_html_report(all_results, aggregated)
        reports["html"] = str(html_path)

        # 4. Generate JSON summary
        json_path = self._save_summary_json(aggregated)
        reports["json"] = str(json_path)

        logger.info(f"Reports generated in {self.results_dir}")
        return reports

    def _save_csv(self, all_results: Dict[str, List[Dict]]) -> Path:
        """Save all results to CSV."""
        rows = []

        for model_name, samples in all_results.items():
            for sample_result in samples:
                row = {
                    "model": model_name,
                    "sample_id": sample_result.get("sample_id", ""),
                    "category": sample_result.get("category", ""),
                    "difficulty": sample_result.get("difficulty", ""),
                }

                # Add metric scores
                metrics = sample_result.get("metrics", {})
                for metric_name, metric_result in metrics.items():
                    if isinstance(metric_result, dict):
                        row[f"metric_{metric_name}"] = metric_result.get("value", 0)
                    else:
                        row[f"metric_{metric_name}"] = metric_result

                # Add judge scores
                judge = sample_result.get("judge", {})
                if isinstance(judge, dict):
                    row["judge_relevance"] = judge.get("relevance", 0)
                    row["judge_accuracy"] = judge.get("accuracy", 0)
                    row["judge_safety"] = judge.get("safety", 0)
                    row["judge_overall"] = judge.get("overall", 0)

                # Add performance metrics
                perf = sample_result.get("performance", {})
                row["latency_seconds"] = perf.get("latency_seconds", 0)
                row["input_tokens"] = perf.get("input_tokens", 0)
                row["output_tokens"] = perf.get("output_tokens", 0)
                row["cost_usd"] = perf.get("cost_usd", 0)

                rows.append(row)

        df = pd.DataFrame(rows)
        csv_path = self.results_dir / f"benchmark_results_{self.timestamp}.csv"
        df.to_csv(csv_path, index=False)

        logger.info(f"Saved {len(rows)} results to {csv_path}")
        return csv_path

    def _generate_leaderboard(self, aggregated: Dict[str, Dict]) -> Path:
        """Generate leaderboard visualization."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib

            matplotlib.use("Agg")  # Non-interactive backend
        except ImportError:
            logger.warning("matplotlib not available, skipping visualization")
            return self.results_dir / "leaderboard.txt"

        # Prepare data
        models = []
        metrics_data = {
            "f1_score": [],
            "cosine_similarity": [],
            "judge_overall": [],
            "latency": [],
            "cost": [],
        }

        for model_name, results in aggregated.items():
            models.append(model_name.replace("_", " ").title())

            for metric in metrics_data.keys():
                if metric == "latency":
                    metrics_data[metric].append(results.get("latency_mean", 0))
                elif metric == "cost":
                    metrics_data[metric].append(results.get("cost_mean", 0))
                else:
                    metrics_data[metric].append(results.get(f"{metric}_mean", 0))

        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("LLM Benchmark Leaderboard", fontsize=16, fontweight="bold")

        colors = plt.cm.Set2(np.linspace(0, 1, len(models)))

        # Quality Metrics
        metric_names = ["f1_score", "cosine_similarity", "judge_overall"]
        for idx, metric in enumerate(metric_names):
            ax = axes[0, idx]
            values = metrics_data[metric]
            bars = ax.barh(models, values, color=colors)
            ax.set_xlabel("Score (0-1)")
            ax.set_title(metric.replace("_", " ").title())
            ax.set_xlim(0, 1)

            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(
                    val + 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}",
                    va="center",
                    fontsize=9,
                )

        # Performance Metrics
        ax = axes[1, 0]
        values = metrics_data["latency"]
        bars = ax.barh(models, values, color=colors)
        ax.set_xlabel("Latency (seconds)")
        ax.set_title("Response Latency")
        for bar, val in zip(bars, values):
            ax.text(
                val + 0.1,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}s",
                va="center",
                fontsize=9,
            )

        # Cost
        ax = axes[1, 1]
        values = metrics_data["cost"]
        bars = ax.barh(models, values, color=colors)
        ax.set_xlabel("Cost (USD)")
        ax.set_title("Cost per 1K samples")
        for bar, val in zip(bars, values):
            ax.text(
                val + 0.001,
                bar.get_y() + bar.get_height() / 2,
                f"${val:.4f}",
                va="center",
                fontsize=9,
            )

        # Overall Score (combined)
        ax = axes[1, 2]
        overall_scores = []
        for i in range(len(models)):
            # Weighted combination: 40% quality, 30% speed, 30% cost efficiency
            quality = (
                metrics_data["f1_score"][i]
                + metrics_data["cosine_similarity"][i]
                + metrics_data["judge_overall"][i]
            ) / 3
            # Normalize latency and cost (inverse, lower is better)
            latency_score = max(0, 1 - metrics_data["latency"][i] / 10)
            cost_score = max(0, 1 - metrics_data["cost"][i] * 1000)
            overall = 0.4 * quality + 0.3 * latency_score + 0.3 * cost_score
            overall_scores.append(overall)

        bars = ax.barh(models, overall_scores, color=colors)
        ax.set_xlabel("Overall Score")
        ax.set_title("Overall Score (Quality + Speed + Cost)")
        ax.set_xlim(0, 1)
        for bar, val in zip(bars, overall_scores):
            ax.text(
                val + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}",
                va="center",
                fontsize=9,
            )

        plt.tight_layout()

        # Save
        leaderboard_path = self.results_dir / f"leaderboard_{self.timestamp}.png"
        plt.savefig(leaderboard_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved leaderboard to {leaderboard_path}")
        return leaderboard_path

    def _generate_html_report(
        self,
        all_results: Dict[str, List[Dict]],
        aggregated: Dict[str, Dict],
    ) -> Path:
        """Generate interactive HTML report."""

        # Build leaderboard table
        leaderboard_rows = []
        for model_name, metrics in sorted(
            aggregated.items(),
            key=lambda x: x[1].get("judge_overall_mean", 0),
            reverse=True,
        ):
            leaderboard_rows.append(f"""
                <tr>
                    <td><strong>{model_name}</strong></td>
                    <td>{metrics.get("f1_score_mean", 0):.3f}</td>
                    <td>{metrics.get("cosine_similarity_mean", 0):.3f}</td>
                    <td>{metrics.get("judge_overall_mean", 0):.2f}/10</td>
                    <td>{metrics.get("latency_mean", 0):.2f}s</td>
                    <td>${metrics.get("cost_mean", 0):.4f}</td>
                </tr>
            """)

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>LLM Benchmark Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
               max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
        h1 {{ color: #1a1a2e; text-align: center; }}
        h2 {{ color: #2563eb; border-bottom: 2px solid #2563eb; padding-bottom: 10px; }}
        table {{ width: 100%; border-collapse: collapse; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #2563eb; color: white; }}
        tr:hover {{ background: #f8f9fa; }}
        .metric-high {{ color: #10b981; font-weight: bold; }}
        .metric-low {{ color: #ef4444; }}
        .card {{ background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .timestamp {{ text-align: center; color: #666; font-size: 0.9em; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }}
        .summary-card {{ background: white; padding: 20px; border-radius: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .summary-value {{ font-size: 2em; font-weight: bold; color: #2563eb; }}
        .summary-label {{ color: #666; margin-top: 5px; }}
    </style>
</head>
<body>
    <h1>LLM Benchmark Report</h1>
    <p class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    
    <div class="card">
        <h2>Leaderboard</h2>
        <table>
            <thead>
                <tr>
                    <th>Model</th>
                    <th>F1 Score</th>
                    <th>Cosine Similarity</th>
                    <th>Judge Score</th>
                    <th>Latency</th>
                    <th>Cost/1K</th>
                </tr>
            </thead>
            <tbody>
                {"".join(leaderboard_rows)}
            </tbody>
        </table>
    </div>
    
    <div class="card">
        <h2>Visual Comparison</h2>
        <p>See <code>leaderboard_{self.timestamp}.png</code> for detailed charts.</p>
    </div>
    
    <div class="card">
        <h2>Best Model Recommendation</h2>
        <p><strong>Best Quality:</strong> {max(aggregated.items(), key=lambda x: x[1].get("judge_overall_mean", 0))[0]}</p>
        <p><strong>Fastest:</strong> {min(aggregated.items(), key=lambda x: x[1].get("latency_mean", float("inf")))[0]}</p>
        <p><strong>Most Cost-Effective:</strong> {min(aggregated.items(), key=lambda x: x[1].get("cost_mean", float("inf")))[0]}</p>
    </div>
</body>
</html>
        """

        html_path = self.results_dir / f"report_{self.timestamp}.html"
        with open(html_path, "w") as f:
            f.write(html_content)

        logger.info(f"Saved HTML report to {html_path}")
        return html_path

    def _save_summary_json(self, aggregated: Dict[str, Dict]) -> Path:
        """Save summary as JSON."""
        summary = {
            "timestamp": self.timestamp,
            "models": list(aggregated.keys()),
            "results": aggregated,
        }

        json_path = self.results_dir / f"summary_{self.timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

        return json_path

    def print_leaderboard(self, aggregated: Dict[str, Dict]):
        """Print a simple text leaderboard."""
        print("\n" + "=" * 80)
        print(" " * 25 + "LLM BENCHMARK RESULTS")
        print("=" * 80)

        # Sort by judge overall score
        sorted_models = sorted(
            aggregated.items(),
            key=lambda x: x[1].get("judge_overall_mean", 0),
            reverse=True,
        )

        print(
            f"\n{'Model':<25} {'F1':>8} {'Cosine':>8} {'Judge':>8} {'Latency':>10} {'Cost':>10}"
        )
        print("-" * 80)

        for rank, (model, metrics) in enumerate(sorted_models, 1):
            print(
                f"{rank}. {model:<21} "
                f"{metrics.get('f1_score_mean', 0):>8.3f} "
                f"{metrics.get('cosine_similarity_mean', 0):>8.3f} "
                f"{metrics.get('judge_overall_mean', 0):>7.2f}/10 "
                f"{metrics.get('latency_mean', 0):>9.2f}s "
                f"${metrics.get('cost_mean', 0):>9.4f}"
            )

        print("=" * 80 + "\n")

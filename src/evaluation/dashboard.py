"""
Rich Dashboard for Console Output

Beautiful terminal UI with tables and progress bars.
"""

from typing import List, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)
from rich.panel import Panel
from rich.text import Text
from rich.live import Live

from .cost import CostTracker, CostSnapshot


class EvaluationDashboard:
    """
    Rich dashboard for real-time evaluation feedback.

    Features:
    - Live progress tracking
    - Model comparison tables
    - Cost monitoring
    - Color-coded results
    """

    def __init__(self):
        self.console = Console()
        self.cost_tracker = CostTracker()
        self._progress = None
        self._task_id = None
        self._results = []

    def print_header(self):
        """Print the dashboard header."""
        header = Text("DeepSeek LLM Evaluation Pipeline", style="bold cyan")
        subtitle = Text("High-Performance Async Evaluation System", style="dim")

        self.console.print(
            Panel.fit(
                f"[cyan]{header}\n{subtitle}[/cyan]",
                border_style="cyan",
                padding=(1, 2),
            )
        )
        self.console.print()

    def create_progress(self) -> Progress:
        """Create a progress bar for evaluation."""
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console,
        )
        return self._progress

    def start_eval(self, model_name: str, total_samples: int) -> int:
        """Start a new evaluation task."""
        if self._progress:
            self._progress.start()
            self._task_id = self._progress.add_task(
                f"[cyan]Evaluating {model_name}[/cyan]",
                total=total_samples,
            )
        return self._task_id

    def update_progress(self, advance: int = 1):
        """Update progress bar."""
        if self._progress and self._task_id is not None:
            self._progress.update(self._task_id, advance=advance)

    def stop_progress(self):
        """Stop the progress bar."""
        if self._progress:
            self._progress.stop()

    def print_model_comparison(self, results: Dict[str, List[Dict]]):
        """Print a comparison table of all models."""
        self.console.print("\n[bold]Model Comparison[/bold]\n")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Avg Score", justify="right")
        table.add_column("Cosine Sim", justify="right")
        table.add_column("Avg Latency", justify="right")
        table.add_column("Total Cost", justify="right")

        for model_name, model_results in results.items():
            if not model_results:
                continue

            # Calculate averages
            scores = [
                r.get("judge_score", 0) for r in model_results if r.get("judge_score")
            ]
            avg_score = sum(scores) / len(scores) if scores else 0

            cos_sims = [
                r.get("metrics", {}).get("cosine_similarity", 0) for r in model_results
            ]
            avg_cos = sum(cos_sims) / len(cos_sims) if cos_sims else 0

            latencies = [r.get("latency_ms", 0) for r in model_results]
            avg_latency = sum(latencies) / len(latencies) if latencies else 0

            costs = [r.get("cost_usd", 0) for r in model_results]
            total_cost = sum(costs)

            # Color code based on score
            score_color = self._get_score_color(avg_score)

            table.add_row(
                model_name,
                f"[{score_color}]{avg_score:.2f}[/{score_color}]",
                f"{avg_cos:.3f}",
                f"{avg_latency:.0f}ms",
                f"${total_cost:.4f}",
            )

        self.console.print(table)
        self.console.print()

    def print_leaderboard(self, results: Dict[str, List[Dict]]):
        """Print a ranked leaderboard."""
        self.console.print("\n[bold yellow]Leaderboard[/bold yellow]\n")

        # Calculate rankings
        rankings = []
        for model_name, model_results in results.items():
            if not model_results:
                continue

            scores = [
                r.get("judge_score", 0) for r in model_results if r.get("judge_score")
            ]
            avg_score = sum(scores) / len(scores) if scores else 0

            rankings.append((model_name, avg_score))

        # Sort by score
        rankings.sort(key=lambda x: x[1], reverse=True)

        medals = ["🥇", "🥈", "🥉"]

        table = Table(show_header=True, header_style="bold green")
        table.add_column("Rank", justify="center", width=6)
        table.add_column("Model", style="cyan")
        table.add_column("Score", justify="right")
        table.add_column("Status", justify="center")

        for i, (model, score) in enumerate(rankings):
            medal = medals[i] if i < 3 else f"#{i + 1}"
            status = "Best" if i == 0 else "Good" if score >= 7 else "Average"
            status_color = (
                "green" if status == "Best" else "yellow" if status == "Good" else "red"
            )

            table.add_row(
                medal,
                model,
                f"{score:.2f}/10",
                f"[{status_color}]{status}[/{status_color}]",
            )

        self.console.print(table)
        self.console.print()

    def print_cost_summary(self, cost_tracker: CostTracker):
        """Print cost breakdown."""
        self.console.print("\n[bold]Cost Summary[/bold]\n")

        table = Table(show_header=True, header_style="bold red")
        table.add_column("Model", style="cyan")
        table.add_column("Requests", justify="right")
        table.add_column("Input Tokens", justify="right")
        table.add_column("Output Tokens", justify="right")
        table.add_column("Total Cost", justify="right")

        for snapshot in cost_tracker.get_all_snapshots():
            table.add_row(
                snapshot.model,
                str(snapshot.total_requests),
                f"{snapshot.total_input_tokens:,}",
                f"{snapshot.total_output_tokens:,}",
                f"${snapshot.total_cost_usd:.4f}",
            )

        self.console.print(table)

        # Total
        total = cost_tracker.total_cost
        self.console.print(f"\n[bold]Total Cost:[/bold] [green]${total:.4f}[/green]")

        if cost_tracker.budget_limit:
            remaining = cost_tracker.budget_limit - total
            self.console.print(f"[bold]Budget Remaining:[/bold] ${remaining:.4f}")

        self.console.print()

    def print_sample_result(self, sample_id: str, model: str, result: Dict):
        """Print a single sample result."""
        score = result.get("judge_score", 0)
        color = self._get_score_color(score)

        self.console.print(
            f"[dim]{sample_id}[/dim] | "
            f"[cyan]{model}[/cyan] | "
            f"[{color}]{score:.1f}/10[/{color}]"
        )

    @staticmethod
    def _get_score_color(score: float) -> str:
        """Get color based on score."""
        if score >= 8:
            return "green"
        elif score >= 6:
            return "yellow"
        else:
            return "red"

    def print_final_report(
        self, results: Dict[str, List[Dict]], cost_tracker: CostTracker
    ):
        """Print the final evaluation report."""
        self.console.print("\n")
        self.print_model_comparison(results)
        self.print_leaderboard(results)
        self.print_cost_summary(cost_tracker)

        # Recommendations
        self.console.print(
            Panel(
                "[bold]Recommendations[/bold]\n\n"
                + self._generate_recommendations(results),
                border_style="green",
                padding=(1, 2),
            )
        )

    def _generate_recommendations(self, results: Dict[str, List[Dict]]) -> str:
        """Generate recommendations based on results."""
        recommendations = []

        # Find best model
        best_model = None
        best_score = 0
        for model, model_results in results.items():
            scores = [
                r.get("judge_score", 0) for r in model_results if r.get("judge_score")
            ]
            avg = sum(scores) / len(scores) if scores else 0
            if avg > best_score:
                best_score = avg
                best_model = model

        if best_model:
            recommendations.append(f"Best overall: [cyan]{best_model}[/cyan]")

        # Find fastest
        fastest_model = None
        fastest_latency = float("inf")
        for model, model_results in results.items():
            latencies = [r.get("latency_ms", 0) for r in model_results]
            avg = sum(latencies) / len(latencies) if latencies else 0
            if avg < fastest_latency and avg > 0:
                fastest_latency = avg
                fastest_model = model

        if fastest_model:
            recommendations.append(f"Fastest: [green]{fastest_model}[/green]")

        return (
            "\n".join(recommendations)
            if recommendations
            else "No recommendations available."
        )

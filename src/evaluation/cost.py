"""
Cost Tracking Module

Real-time cost calculation and budget tracking.
"""

from dataclasses import dataclass
from typing import Dict, List
from collections import defaultdict

from .config import DEEPSEEK_PRICING, ModelResponse


@dataclass
class CostSnapshot:
    """Snapshot of costs at a point in time."""

    model: str
    total_requests: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float

    @property
    def avg_cost_per_request(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_cost_usd / self.total_requests


class CostTracker:
    """
    Track costs across all API calls.

    Provides real-time monitoring and budget alerts.
    """

    def __init__(self, budget_limit: float = None):
        """
        Initialize cost tracker.

        Args:
            budget_limit: Optional budget cap in USD
        """
        self.budget_limit = budget_limit
        self._requests: Dict[str, List] = defaultdict(list)
        self._costs: Dict[str, float] = defaultdict(float)
        self._tokens: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"input": 0, "output": 0}
        )
        self._request_counts: Dict[str, int] = defaultdict(int)

    def record(self, model: str, response: ModelResponse):
        """Record a single API call."""
        self._requests[model].append(response)
        self._tokens[model]["input"] += response.input_tokens
        self._tokens[model]["output"] += response.output_tokens
        self._costs[model] += response.cost_usd
        self._request_counts[model] += 1

        # Check budget
        if self.budget_limit and self.total_cost > self.budget_limit:
            raise BudgetExceededError(
                f"Budget limit exceeded: ${self.total_cost:.4f} > ${self.budget_limit:.4f}"
            )

    def record_batch(self, model: str, responses: List[ModelResponse]):
        """Record multiple API calls."""
        for response in responses:
            self.record(model, response)

    @property
    def total_cost(self) -> float:
        """Total cost across all models."""
        return sum(self._costs.values())

    @property
    def total_tokens(self) -> Dict[str, int]:
        """Total tokens across all models."""
        return {
            "input": sum(t["input"] for t in self._tokens.values()),
            "output": sum(t["output"] for t in self._tokens.values()),
        }

    @property
    def total_requests(self) -> int:
        """Total number of requests."""
        return sum(self._request_counts.values())

    def get_snapshot(self, model: str = None) -> CostSnapshot:
        """Get cost snapshot for a model or all models."""
        if model:
            return CostSnapshot(
                model=model,
                total_requests=self._request_counts[model],
                total_input_tokens=self._tokens[model]["input"],
                total_output_tokens=self._tokens[model]["output"],
                total_cost_usd=self._costs[model],
            )
        else:
            return CostSnapshot(
                model="all",
                total_requests=self.total_requests,
                total_input_tokens=self.total_tokens["input"],
                total_output_tokens=self.total_tokens["output"],
                total_cost_usd=self.total_cost,
            )

    def get_all_snapshots(self) -> List[CostSnapshot]:
        """Get snapshots for all models."""
        return [self.get_snapshot(model) for model in self._costs.keys()]

    def estimate_batch_cost(
        self,
        model: str,
        num_requests: int,
        avg_input_tokens: int,
        avg_output_tokens: int,
    ) -> float:
        """Estimate cost for a batch of requests."""
        pricing = DEEPSEEK_PRICING.get(model)
        if not pricing:
            return 0.0

        return pricing.calculate_cost(
            num_requests * avg_input_tokens,
            num_requests * avg_output_tokens,
        )

    def format_cost(self, cost: float) -> str:
        """Format cost for display."""
        if cost < 0.001:
            return f"${cost * 1000:.2f}m"
        elif cost < 1:
            return f"${cost:.4f}"
        else:
            return f"${cost:.2f}"

    def summary(self) -> Dict:
        """Get a summary dictionary."""
        return {
            "total_cost": self.total_cost,
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "by_model": {
                model: {
                    "cost": self._costs[model],
                    "requests": self._request_counts[model],
                    "input_tokens": self._tokens[model]["input"],
                    "output_tokens": self._tokens[model]["output"],
                }
                for model in self._costs.keys()
            },
            "budget_remaining": (
                self.budget_limit - self.total_cost if self.budget_limit else None
            ),
        }


class BudgetExceededError(Exception):
    """Raised when budget limit is exceeded."""

    pass

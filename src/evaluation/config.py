"""
Configuration and Pricing for DeepSeek Evaluation Pipeline
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class ModelType(Enum):
    """DeepSeek model types."""

    CHAT = "deepseek-chat"
    REASONER = "deepseek-reasoner"


@dataclass
class ModelPricing:
    """Pricing for a DeepSeek model (per 1M tokens)."""

    input_cost: float  # $ per 1M input tokens
    output_cost: float  # $ per 1M output tokens

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate total cost in USD."""
        return (input_tokens / 1_000_000) * self.input_cost + (
            output_tokens / 1_000_000
        ) * self.output_cost


# DeepSeek 2026 Pricing
DEEPSEEK_PRICING = {
    "deepseek-chat": ModelPricing(input_cost=0.27, output_cost=1.10),  # V3
    "deepseek-reasoner": ModelPricing(input_cost=0.55, output_cost=2.19),  # R1
}

# Default models to evaluate
DEFAULT_MODELS = [
    {
        "name": "DeepSeek V3",
        "model_id": "deepseek-chat",
        "type": ModelType.CHAT,
        "description": "Fast chat model for general tasks",
    },
    {
        "name": "DeepSeek R1",
        "model_id": "deepseek-reasoner",
        "type": ModelType.REASONER,
        "description": "Chain-of-thought reasoning model",
    },
]


@dataclass
class EvalConfig:
    """Evaluation configuration."""

    api_key: str = ""
    base_url: str = "https://api.deepseek.com"
    max_concurrent: int = 10
    timeout: int = 120
    max_tokens: int = 4096
    temperature: float = 0.7
    save_results: bool = True
    results_dir: str = "results"

    def __post_init__(self):
        if not self.api_key:
            self.api_key = os.environ.get("DEEPSEEK_API_KEY", "")

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)


@dataclass
class EvalSample:
    """Single evaluation sample (Golden Set)."""

    id: str
    question: str
    context: str
    ground_truth: str
    category: str = "general"
    difficulty: str = "medium"

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "question": self.question,
            "context": self.context,
            "ground_truth": self.ground_truth,
            "category": self.category,
            "difficulty": self.difficulty,
        }


@dataclass
class ModelResponse:
    """Response from a model."""

    content: str
    reasoning: Optional[str] = None  # Only for R1
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0
    cost_usd: float = 0
    error: Optional[str] = None


@dataclass
class EvalResult:
    """Result of evaluating one sample."""

    sample_id: str
    model_name: str
    response: ModelResponse
    metrics: Dict[str, float] = field(default_factory=dict)
    judge_score: Optional[float] = None
    judge_reasoning: Optional[str] = None

"""
DeepSeek R1 LLM-as-Judge

Uses DeepSeek R1's chain-of-thought reasoning for high-quality evaluation.
"""

import asyncio
import logging
from typing import Dict, Optional
from dataclasses import dataclass

from .async_client import DeepSeekAsyncClient
from .config import EvalConfig

logger = logging.getLogger(__name__)


@dataclass
class JudgeResult:
    """Result from LLM-as-Judge evaluation."""

    score: float  # 0-10
    reasoning: str  # Chain-of-thought reasoning
    feedback: str  # Detailed feedback
    hallucinations: float  # 0-1 (hallucination risk)

    def to_dict(self) -> Dict:
        return {
            "score": self.score,
            "reasoning": self.reasoning,
            "feedback": self.feedback,
            "hallucinations": self.hallucinations,
        }


class DeepSeekJudge:
    """
    LLM-as-Judge using DeepSeek R1.

    R1 provides superior evaluation due to its chain-of-thought reasoning.
    The reasoning content is captured and logged separately for transparency.
    """

    SYSTEM_PROMPT = """You are an expert legal AI evaluator with exceptional analytical capabilities.

Your task is to evaluate responses to legal questions on a scale of 0-10:

EVALUATION CRITERIA:
1. RELEVANCE (0-10): Does the response directly address the question?
2. ACCURACY (0-10): Is the response factually correct based on the context?
3. COMPLETENESS (0-10): Does it cover all aspects of the question?
4. SAFETY (0-10): Does it avoid hallucinations and harmful content?

IMPORTANT INSTRUCTIONS:
- Be strict but fair in your evaluation
- Legal answers must be precise and cite relevant details
- Consider the difficulty of the question
- A score of 10 means perfect, 5 means average, 0 means failing

Output your evaluation as JSON with this exact format:
{
    "relevance": [0-10],
    "accuracy": [0-10],
    "completeness": [0-10],
    "safety": [0-10],
    "overall": [average of above],
    "hallucination_risk": [0-1, where 1 = high risk],
    "reasoning": "[your step-by-step reasoning]",
    "feedback": "[constructive feedback for improvement]"
}"""

    def __init__(self, client: DeepSeekAsyncClient):
        self.client = client

    async def evaluate(
        self,
        question: str,
        context: str,
        response: str,
        ground_truth: str,
    ) -> JudgeResult:
        """
        Evaluate a response using DeepSeek R1 as judge.

        Args:
            question: The original question
            context: Source context
            response: Model's response to evaluate
            ground_truth: Expected correct answer

        Returns:
            JudgeResult with scores and chain-of-thought reasoning
        """
        messages = [
            {
                "role": "user",
                "content": self._build_prompt(
                    question, context, response, ground_truth
                ),
            }
        ]

        model_response = await self.client.chat(
            model="deepseek-reasoner",
            messages=messages,
            system_prompt=self.SYSTEM_PROMPT,
        )

        if model_response.error:
            logger.error(f"Judge error: {model_response.error}")
            return JudgeResult(
                score=0,
                reasoning="Evaluation failed",
                feedback=f"Error: {model_response.error}",
                hallucinations=1.0,
            )

        # Parse the JSON response
        try:
            import json
            import re

            # Extract JSON from response
            json_match = re.search(
                r'\{[^{}]*"overall"[^{}]*\}', model_response.content, re.DOTALL
            )
            if json_match:
                data = json.loads(json_match.group())
            else:
                # Try to find any JSON object
                start = model_response.content.find("{")
                end = model_response.content.rfind("}") + 1
                if start != -1 and end > start:
                    data = json.loads(model_response.content[start:end])
                else:
                    raise ValueError("No JSON found in response")

            # Log the reasoning separately (R1's strength!)
            logger.info(f"Judge reasoning:\n{model_response.reasoning or 'N/A'}")

            return JudgeResult(
                score=float(data.get("overall", 5)),
                reasoning=model_response.reasoning or data.get("reasoning", ""),
                feedback=data.get("feedback", ""),
                hallucinations=float(data.get("hallucination_risk", 0.5)),
            )

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse judge response: {e}")
            # Estimate score from content
            content_lower = model_response.content.lower()
            if "10" in content_lower or "perfect" in content_lower:
                estimated_score = 9.0
            elif "5" in content_lower or "average" in content_lower:
                estimated_score = 5.0
            else:
                estimated_score = 7.0

            return JudgeResult(
                score=estimated_score,
                reasoning=model_response.reasoning or "Parse error - estimated",
                feedback=model_response.content[:200],
                hallucinations=0.3,
            )

    async def batch_evaluate(
        self,
        evaluations: list,
        max_concurrent: int = 3,  # Limit for judge to ensure quality
    ) -> list:
        """
        Evaluate multiple responses in batch.

        Note: Lower concurrency for judge to ensure consistent quality.
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_evaluate(eval_data):
            async with semaphore:
                return await self.evaluate(
                    question=eval_data["question"],
                    context=eval_data.get("context", ""),
                    response=eval_data["response"],
                    ground_truth=eval_data["ground_truth"],
                )

        tasks = [bounded_evaluate(e) for e in evaluations]
        return await asyncio.gather(*tasks)

    @staticmethod
    def _build_prompt(
        question: str,
        context: str,
        response: str,
        ground_truth: str,
    ) -> str:
        """Build the evaluation prompt."""
        return f"""EVALUATE THIS RESPONSE:

QUESTION: {question}

SOURCE CONTEXT:
{context}

MODEL RESPONSE TO EVALUATE:
{response}

EXPECTED ANSWER (Ground Truth):
{ground_truth}

Provide your evaluation in JSON format with your step-by-step reasoning."""

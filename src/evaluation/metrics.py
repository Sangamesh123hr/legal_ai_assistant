"""
Evaluation Metrics for LLM Responses
"""

import re
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """Result from a single metric evaluation."""

    name: str
    value: float
    details: str = ""

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "value": self.value,
            "details": self.details,
        }


class MetricsCalculator:
    """Calculate evaluation metrics for LLM responses."""

    def __init__(self, embedder_api_key: str = None):
        self.embedder_api_key = embedder_api_key
        self._embedder = None

    def calculate_all(
        self, prediction: str, ground_truth: str, context: str = ""
    ) -> List[MetricResult]:
        """
        Calculate all available metrics.

        Args:
            prediction: Model's predicted answer
            ground_truth: Expected correct answer
            context: Optional context (for RAG evaluation)

        Returns:
            List of MetricResult objects
        """
        results = []

        # Exact Match
        results.append(self.exact_match(prediction, ground_truth))

        # F1 Score
        results.append(self.f1_score(prediction, ground_truth))

        # Cosine Similarity
        results.append(self.cosine_similarity(prediction, ground_truth))

        # ROUGE-like metrics
        results.append(self.bleu_approx(prediction, ground_truth))

        # Length ratio
        results.append(self.length_ratio(prediction, ground_truth))

        return results

    def exact_match(self, prediction: str, ground_truth: str) -> MetricResult:
        """
        Exact match metric - binary score for identical strings.
        Normalized to handle minor variations.
        """
        # Normalize strings
        pred_norm = self._normalize(prediction)
        truth_norm = self._normalize(ground_truth)

        match = pred_norm == truth_norm
        score = 1.0 if match else 0.0

        return MetricResult(name="exact_match", value=score, details=f"Match: {match}")

    def f1_score(self, prediction: str, ground_truth: str) -> MetricResult:
        """
        Token-level F1 score between prediction and ground truth.
        """
        pred_tokens = set(self._tokenize(prediction))
        truth_tokens = set(self._tokenize(ground_truth))

        if not pred_tokens and not truth_tokens:
            return MetricResult(name="f1_score", value=1.0, details="Empty strings")

        if not pred_tokens or not truth_tokens:
            return MetricResult(name="f1_score", value=0.0, details="One empty string")

        # Calculate overlap
        overlap = pred_tokens & truth_tokens
        precision = len(overlap) / len(pred_tokens) if pred_tokens else 0
        recall = len(overlap) / len(truth_tokens) if truth_tokens else 0

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        return MetricResult(
            name="f1_score", value=f1, details=f"P={precision:.2f}, R={recall:.2f}"
        )

    def cosine_similarity(self, prediction: str, ground_truth: str) -> MetricResult:
        """
        Cosine similarity using TF-IDF or word embeddings.
        Falls back to word overlap if embeddings unavailable.
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([prediction, ground_truth])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

            return MetricResult(
                name="cosine_similarity",
                value=float(similarity),
                details="TF-IDF based",
            )
        except ImportError:
            logger.warning("sklearn not available, using word overlap")
            return self._word_overlap_similarity(prediction, ground_truth)

    def _word_overlap_similarity(
        self, prediction: str, ground_truth: str
    ) -> MetricResult:
        """Fallback: word overlap similarity."""
        pred_words = set(self._tokenize(prediction.lower()))
        truth_words = set(self._tokenize(ground_truth.lower()))

        if not pred_words and not truth_words:
            return MetricResult(
                name="cosine_similarity", value=1.0, details="Word overlap"
            )

        if not pred_words or not truth_words:
            return MetricResult(
                name="cosine_similarity", value=0.0, details="Word overlap"
            )

        overlap = pred_words & truth_words
        similarity = 2 * len(overlap) / (len(pred_words) + len(truth_words))

        return MetricResult(
            name="cosine_similarity", value=similarity, details="Word overlap based"
        )

    def bleu_approx(self, prediction: str, ground_truth: str) -> MetricResult:
        """
        Simplified BLEU-like metric using n-gram precision.
        """

        def get_ngrams(tokens: List[str], n: int) -> Dict[Tuple, int]:
            return {tuple(tokens[i : i + n]): 1 for i in range(len(tokens) - n + 1)}

        pred_tokens = self._tokenize(prediction)
        truth_tokens = self._tokenize(ground_truth)

        if len(pred_tokens) < 2 or len(truth_tokens) < 2:
            return MetricResult(name="bleu_approx", value=0.0, details="Too short")

        # Calculate 1-gram and 2-gram precision
        pred_unigrams = get_ngrams(pred_tokens, 1)
        truth_unigrams = get_ngrams(truth_tokens, 1)
        pred_bigrams = get_ngrams(pred_tokens, 2)
        truth_bigrams = get_ngrams(truth_tokens, 2)

        # Unigram precision
        unigram_overlap = len(set(pred_unigrams.keys()) & set(truth_unigrams.keys()))
        p1 = unigram_overlap / len(pred_unigrams) if pred_unigrams else 0

        # Bigram precision
        bigram_overlap = len(set(pred_bigrams.keys()) & set(truth_bigrams.keys()))
        p2 = bigram_overlap / len(pred_bigrams) if pred_bigrams else 0

        # Simplified BLEU
        bleu = np.exp(0.5 * np.log(p1 + 1e-10) + 0.5 * np.log(p2 + 1e-10))

        return MetricResult(
            name="bleu_approx", value=float(bleu), details=f"P1={p1:.2f}, P2={p2:.2f}"
        )

    def length_ratio(self, prediction: str, ground_truth: str) -> MetricResult:
        """Check if response length is reasonable."""
        pred_len = len(prediction)
        truth_len = len(ground_truth)

        if truth_len == 0:
            ratio = 1.0 if pred_len == 0 else 0.0
        else:
            ratio = min(pred_len, truth_len) / max(pred_len, truth_len)

        return MetricResult(
            name="length_ratio",
            value=ratio,
            details=f"Pred: {pred_len}, Truth: {truth_len}, Ratio: {ratio:.2f}",
        )

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        # Lowercase
        text = text.lower()
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        # Remove punctuation (optional)
        text = re.sub(r"[^\w\s]", "", text)
        return text

    def _tokenize(self, text: str) -> List[str]:
        """Simple word tokenization."""
        return self._normalize(text).split()

    def aggregate_results(self, results: List[List[MetricResult]]) -> Dict[str, Dict]:
        """
        Aggregate results from multiple evaluations.

        Args:
            results: List of metric result lists (one per sample)

        Returns:
            Dictionary with aggregated statistics
        """
        if not results:
            return {}

        # Collect all metric names
        metric_names = set()
        for sample_results in results:
            for result in sample_results:
                metric_names.add(result.name)

        # Calculate aggregates
        aggregates = {}
        for metric_name in metric_names:
            values = [
                r.value
                for sample_results in results
                for r in sample_results
                if r.name == metric_name
            ]

            if values:
                aggregates[metric_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "median": np.median(values),
                    "count": len(values),
                }

        return aggregates

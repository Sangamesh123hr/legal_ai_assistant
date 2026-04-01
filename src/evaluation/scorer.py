"""
Local Embedding Scorer

Uses local sentence-transformers for semantic similarity scoring.
No API costs - runs entirely on CPU/GPU.
"""

import logging
from typing import List, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class LocalEmbeddingScorer:
    """
    Local semantic similarity scorer using sentence-transformers.

    Benefits:
    - No API costs
    - Runs offline
    - Fast inference
    - Private (no data leaves machine)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._embedding_dim = 384

    @property
    def model(self):
        """Lazy-load the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info(f"Loading embedding model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
                self._embedding_dim = self._model.get_sentence_embedding_dimension()
                logger.info(f"Model loaded. Embedding dimension: {self._embedding_dim}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )
        return self._model

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to embeddings.

        Args:
            texts: List of text strings

        Returns:
            Numpy array of embeddings (num_texts, embedding_dim)
        """
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def cosine_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts.

        Returns:
            Similarity score from 0 (different) to 1 (identical)
        """
        embeddings = self.encode([text1, text2])
        return self._cosine_sim(embeddings[0], embeddings[1])

    def batch_similarity(
        self,
        predictions: List[str],
        ground_truths: List[str],
    ) -> List[float]:
        """
        Calculate similarity for multiple prediction-ground_truth pairs.

        Returns:
            List of similarity scores
        """
        all_texts = predictions + ground_truths
        embeddings = self.encode(all_texts)

        pred_embeddings = embeddings[: len(predictions)]
        truth_embeddings = embeddings[len(predictions) :]

        # Calculate cosine similarity for each pair
        similarities = []
        for pred_emb, truth_emb in zip(pred_embeddings, truth_embeddings):
            sim = self._cosine_sim(pred_emb, truth_emb)
            similarities.append(float(sim))

        return similarities

    def semantic_f1(
        self,
        prediction: str,
        ground_truth: str,
    ) -> float:
        """
        Calculate semantic F1 score.

        Based on embedding similarity with token overlap heuristics.
        """
        pred_tokens = set(prediction.lower().split())
        truth_tokens = set(ground_truth.lower().split())

        # Token overlap
        overlap = pred_tokens & truth_tokens
        precision = len(overlap) / len(pred_tokens) if pred_tokens else 0
        recall = len(overlap) / len(truth_tokens) if truth_tokens else 0

        if precision + recall == 0:
            token_f1 = 0.0
        else:
            token_f1 = 2 * precision * recall / (precision + recall)

        # Semantic similarity
        sem_sim = self.cosine_similarity(prediction, ground_truth)

        # Combined score (50% token, 50% semantic)
        return 0.5 * token_f1 + 0.5 * sem_sim

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))

    def calculate_metrics(
        self,
        prediction: str,
        ground_truth: str,
    ) -> dict:
        """
        Calculate all semantic metrics for a prediction.

        Returns:
            Dictionary with metric_name: score pairs
        """
        return {
            "cosine_similarity": self.cosine_similarity(prediction, ground_truth),
            "semantic_f1": self.semantic_f1(prediction, ground_truth),
        }

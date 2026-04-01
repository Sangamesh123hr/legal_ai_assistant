"""
Dataset Loader for Evaluation

Loads golden sets for evaluation.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

from .config import EvalSample

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Load and manage evaluation datasets."""

    def __init__(self, dataset_path: str = None):
        self.dataset_path = dataset_path
        self._samples: List[EvalSample] = []

    def load(self, limit: Optional[int] = None) -> List[EvalSample]:
        """Load samples from file or create sample data."""
        if self.dataset_path and Path(self.dataset_path).exists():
            return self._load_from_file(limit)
        else:
            return self._create_sample_data(limit)

    def _load_from_file(self, limit: Optional[int] = None) -> List[EvalSample]:
        """Load from JSONL file."""
        samples = []
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                data = json.loads(line.strip())
                samples.append(EvalSample(**data))

        logger.info(f"Loaded {len(samples)} samples from {self.dataset_path}")
        return samples

    def _create_sample_data(self, limit: Optional[int] = None) -> List[EvalSample]:
        """Create sample legal QA dataset."""
        samples = [
            EvalSample(
                id="legal_001",
                question="What are the payment terms in this contract?",
                context="ARTICLE 2: PAYMENT TERMS 2.1 Total: $150,000. 30% ($45,000) on signing. 40% ($60,000) at milestone. 30% ($45,000) at delivery.",
                ground_truth="Payment: Total $150,000. 30% on signing ($45,000), 40% at milestone ($60,000), 30% at delivery ($45,000).",
                category="contract_review",
                difficulty="easy",
            ),
            EvalSample(
                id="legal_002",
                question="What is the termination clause?",
                context="ARTICLE 5: TERMINATION 5.1 Either party may terminate for material breach. 5.2 Client may terminate with 30 days written notice. 5.3 All completed work must be delivered upon termination.",
                ground_truth="Termination: Either party for material breach, or client with 30 days notice. Completed work must be delivered.",
                category="contract_review",
                difficulty="easy",
            ),
            EvalSample(
                id="legal_003",
                question="Who owns the intellectual property?",
                context="ARTICLE 3: IP 3.1 All IP rights transfer to Client upon final payment. 3.2 Developer retains right to use general knowledge gained.",
                ground_truth="IP transfers to Client on final payment. Developer keeps rights to general knowledge.",
                category="contract_review",
                difficulty="medium",
            ),
            EvalSample(
                id="legal_004",
                question="What are the confidentiality requirements?",
                context="ARTICLE 4: CONFIDENTIALITY 4.1 Both parties protect confidential info. 4.2 3 years after termination. 4.3 Excludes public domain info.",
                ground_truth="Confidentiality: Both parties protect info for 3 years post-termination. Excludes public domain.",
                category="compliance_check",
                difficulty="medium",
            ),
            EvalSample(
                id="legal_005",
                question="What are the key risk factors?",
                context="RISKS: No liability cap. Unlimited indemnification. No insurance required. Auto-renewal with 60-day notice. No dispute resolution specified.",
                ground_truth="Risks: No liability cap, unlimited indemnification, no insurance, auto-renewal (60-day notice), no dispute resolution.",
                category="risk_identification",
                difficulty="hard",
            ),
            EvalSample(
                id="legal_006",
                question="What is the delivery timeline?",
                context="SCOPE: Custom CRM software. 90 days from Effective Date. 14-day acceptance testing after delivery.",
                ground_truth="Timeline: 90 days for CRM delivery, then 14-day acceptance testing.",
                category="contract_review",
                difficulty="easy",
            ),
            EvalSample(
                id="legal_007",
                question="Are there non-compete clauses?",
                context="NON-COMPETE: Developer cannot work with competitors for 12 months. Geographic limit: India only.",
                ground_truth="Non-compete: 12-month restriction post-completion, India only.",
                category="contract_review",
                difficulty="medium",
            ),
            EvalSample(
                id="legal_008",
                question="What dispute resolution is specified?",
                context="DISPUTES: First mediation. Then ICC arbitration in Singapore. Each party bears own legal costs.",
                ground_truth="Dispute resolution: Mediation first, then ICC arbitration in Singapore. Each party pays own costs.",
                category="compliance_check",
                difficulty="medium",
            ),
        ]

        if limit:
            samples = samples[:limit]

        logger.info(f"Created {len(samples)} sample questions")
        return samples

"""
Pydantic schemas for API requests/responses
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class AnalysisType(str, Enum):
    CONTRACT_REVIEW = "contract_review"
    CLAUSE_EXTRACTION = "clause_extraction"
    RISK_IDENTIFICATION = "risk_identification"
    COMPLIANCE_CHECK = "compliance_check"
    GENERAL = "general"


class AnalyzeRequest(BaseModel):
    document_text: str = Field(..., min_length=10)
    analysis_type: AnalysisType = Field(default=AnalysisType.GENERAL)
    question: Optional[str] = None


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2)
    k: int = Field(default=3, ge=1, le=10)


class BatchAnalyzeRequest(BaseModel):
    documents: List[str] = Field(..., min_items=1, max_items=50)
    analysis_type: AnalysisType = Field(default=AnalysisType.GENERAL)


class Citation(BaseModel):
    source: str
    page: Optional[int] = None
    relevance: float


class AnalysisResponse(BaseModel):
    success: bool
    analysis: str
    citations: List[Citation] = []
    metadata: dict = {}


class QueryResult(BaseModel):
    content: str
    source: str
    score: float


class QueryResponse(BaseModel):
    success: bool
    query: str
    results: List[QueryResult] = []
    total_results: int


class BatchAnalyzeResponse(BaseModel):
    success: bool
    total: int
    successful: int
    failed: int
    results: List[dict]

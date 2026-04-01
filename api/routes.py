"""
API routes for Legal AI Assistant
"""

import logging
from typing import List
from fastapi import APIRouter, HTTPException, UploadFile, File

from api.schemas import (
    AnalyzeRequest,
    AnalysisResponse,
    AnalysisType,
    QueryRequest,
    QueryResponse,
    QueryResult,
    BatchAnalyzeRequest,
    BatchAnalyzeResponse,
    Citation,
)
from api.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Import existing RAG components
try:
    from src.rag.pipeline import rag_pipeline

    RAG_AVAILABLE = True
except ImportError:
    logger.warning("RAG components not available")
    RAG_AVAILABLE = False
    rag_pipeline = None


@router.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_document(request: AnalyzeRequest):
    """Analyze a legal document with optional RAG query."""
    if not RAG_AVAILABLE:
        raise HTTPException(status_code=503, detail="RAG pipeline not available")

    try:
        if request.question:
            result = rag_pipeline.query(request.question, k=3)
            if result.get("success"):
                citations = [
                    Citation(source=r["source"], relevance=r.get("score", 0.8))
                    for r in result.get("results", [])
                ]
                return AnalysisResponse(
                    success=True,
                    analysis=result.get("results", [{}])[0].get(
                        "content", "No content"
                    ),
                    citations=citations,
                    metadata={"analysis_type": request.analysis_type.value},
                )

        return AnalysisResponse(
            success=True,
            analysis=f"Analysis type: {request.analysis_type.value}. Document length: {len(request.document_text)} chars.",
            citations=[],
            metadata={"analysis_type": request.analysis_type.value},
        )
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=QueryResponse, tags=["Knowledge Base"])
async def query_knowledge_base(request: QueryRequest):
    """Query the legal knowledge base using RAG."""
    if not RAG_AVAILABLE:
        raise HTTPException(status_code=503, detail="RAG pipeline not available")

    try:
        result = rag_pipeline.query(request.query, k=request.k)

        results = [
            QueryResult(
                content=r["content"], source=r["source"], score=r.get("score", 0.8)
            )
            for r in result.get("results", [])
        ]

        return QueryResponse(
            success=result.get("success", False),
            query=request.query,
            results=results,
            total_results=len(results),
        )
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=BatchAnalyzeResponse, tags=["Batch"])
async def batch_analyze(request: BatchAnalyzeRequest):
    """Analyze multiple documents in batch."""
    results, successful, failed = [], 0, 0

    for doc in request.documents:
        try:
            results.append({"document": doc[:50] + "...", "success": True})
            successful += 1
        except Exception as e:
            results.append(
                {"document": doc[:50] + "...", "success": False, "error": str(e)}
            )
            failed += 1

    return BatchAnalyzeResponse(
        success=True,
        total=len(request.documents),
        successful=successful,
        failed=failed,
        results=results,
    )


@router.post("/ingest", tags=["Knowledge Base"])
async def ingest_document(file: UploadFile = File(...)):
    """Ingest a document into the knowledge base."""
    if not RAG_AVAILABLE:
        raise HTTPException(status_code=503, detail="RAG pipeline not available")

    if not any(file.filename.endswith(ext) for ext in settings.ALLOWED_EXTENSIONS):
        raise HTTPException(status_code=400, detail=f"File type not allowed")

    try:
        content = await file.read()
        text = content.decode("utf-8", errors="ignore")

        import tempfile, os

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=file.filename, delete=False
        ) as f:
            f.write(text)
            temp_path = f.name

        try:
            result = rag_pipeline.ingest_document(temp_path)
            return {
                "success": result.get("success", False),
                "file": file.filename,
                "chunks": result.get("chunks", 0),
            }
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    except Exception as e:
        logger.error(f"Ingest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status", tags=["System"])
async def get_status():
    """Get system status and statistics."""
    if not RAG_AVAILABLE:
        return {"status": "limited", "rag_available": False}

    try:
        status = rag_pipeline.get_system_status()
        return {
            "status": "operational",
            "rag_available": True,
            "statistics": status.get("statistics", {}),
        }
    except Exception as e:
        return {"status": "error", "rag_available": False, "error": str(e)}

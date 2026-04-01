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


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file using PyPDF2."""
    try:
        from PyPDF2 import PdfReader

        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")


def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX file."""
    try:
        from docx import Document

        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        logger.error(f"DOCX extraction error: {e}")
        raise ValueError(f"Failed to extract text from DOCX: {str(e)}")


def extract_text_from_file(file_path: str, filename: str) -> str:
    """Extract text from any supported file type."""
    ext = filename.lower().split(".")[-1]

    if ext == "pdf":
        return extract_text_from_pdf(file_path)
    elif ext == "docx":
        return extract_text_from_docx(file_path)
    elif ext in ["txt", "md"]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file type: {ext}")


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
    """Ingest a document (PDF, DOCX, TXT) into the knowledge base."""
    if not RAG_AVAILABLE:
        raise HTTPException(status_code=503, detail="RAG pipeline not available")

    if not any(
        file.filename.lower().endswith(ext) for ext in settings.ALLOWED_EXTENSIONS
    ):
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Supported: {settings.ALLOWED_EXTENSIONS}",
        )

    try:
        import tempfile, os

        # Save uploaded file temporarily
        content = await file.read()

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(content)
            temp_path = tmp.name

        try:
            # Extract text based on file type
            file_ext = file.filename.lower().split(".")[-1]

            if file_ext == "pdf":
                text = extract_text_from_pdf(temp_path)
            elif file_ext == "docx":
                text = extract_text_from_docx(temp_path)
            else:
                # For txt/md files, read directly
                with open(temp_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()

            if not text.strip():
                return {
                    "success": False,
                    "file": file.filename,
                    "error": "No text extracted from document",
                }

            # Save extracted text as temp file for ingestion
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as txt_tmp:
                txt_tmp.write(text)
                txt_path = txt_tmp.name

            try:
                result = rag_pipeline.ingest_document(txt_path)
                return {
                    "success": result.get("success", False),
                    "file": file.filename,
                    "chunks": result.get("chunks", 0),
                    "text_length": len(text),
                    "message": f"Successfully extracted {len(text)} characters and ingested {result.get('chunks', 0)} chunks",
                }
            finally:
                if os.path.exists(txt_path):
                    os.remove(txt_path)

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Ingest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status", tags=["System"])
async def get_status():
    """Get system status and statistics."""
    if not RAG_AVAILABLE:
        return {
            "status": "limited",
            "rag_available": False,
            "message": "RAG components not initialized",
        }

    try:
        status = rag_pipeline.get_system_status()
        return {
            "status": "operational",
            "rag_available": True,
            "statistics": status.get("statistics", {}),
        }
    except Exception as e:
        return {"status": "error", "rag_available": False, "error": str(e)}

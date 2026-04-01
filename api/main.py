"""
Production FastAPI Application for Legal AI Assistant
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time

from api.routes import router as api_router
from api.middleware import setup_middleware
from api.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Request counter for metrics
request_count = 0
error_count = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for startup/shutdown."""
    logger.info("Starting Legal AI Assistant API...")
    yield
    logger.info("Shutting down Legal AI Assistant API...")


# Create FastAPI app
app = FastAPI(
    title="Legal AI Assistant API",
    description="Production-grade legal document analysis with multi-agent orchestration",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Track request metrics."""
    global request_count
    request_count += 1
    start_time = time.time()
    try:
        response = await call_next(request)
        return response
    finally:
        duration = time.time() - start_time
        logger.info(f"{request.method} {request.url.path} - {duration:.3f}s")


@app.get("/", tags=["Root"])
async def root():
    return {"name": "Legal AI Assistant API", "version": "1.0.0", "status": "running"}


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy", "service": "legal-ai-assistant", "version": "1.0.0"}


@app.get("/metrics", tags=["Metrics"])
async def metrics():
    return {
        "requests_total": request_count,
        "errors_total": error_count,
        "environment": settings.ENVIRONMENT,
    }


# Setup custom middleware
setup_middleware(app)

# Include API routes
app.include_router(api_router, prefix="/api/v1")


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"error": "Internal server error"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=settings.DEBUG)

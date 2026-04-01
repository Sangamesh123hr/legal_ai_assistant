# Legal AI Assistant

> Production-grade legal document analysis with multi-agent orchestration and RAG pipeline

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Production-green.svg)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-purple.svg)](https://langchain.dev)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)

## Overview

A production-ready legal AI system that analyzes contracts, extracts clauses, identifies risks, and answers legal questions using:

- **Multi-Agent Orchestration** (Analyst → Critic → Judge)
- **RAG Pipeline** with FAISS vector database
- **Structured Outputs** with Pydantic validation
- **Production API** with rate limiting & monitoring

## Key Features

| Feature | Description |
|---------|-------------|
| Document Analysis | Contract review, clause extraction, risk identification |
| RAG Pipeline | Semantic search with citations |
| Multi-Agent | LangGraph orchestration with Claude + DeepSeek |
| Production API | FastAPI with rate limiting, monitoring, health checks |

## Quick Start

### Local Development

```bash
pip install -r requirements-prod.txt
cp .env.example .env
# Edit .env with your API keys
uvicorn api.main:app --reload
# Open http://localhost:8000/docs
```

### Docker Deployment

```bash
docker-compose up -d
curl http://localhost:8000/health
```

## Project Structure

```
legal-ai-api/
├── api/                    # Production FastAPI application
│   ├── main.py            # FastAPI app with middleware
│   ├── routes.py          # API endpoints
│   ├── schemas.py         # Pydantic models
│   ├── config.py          # Settings
│   └── middleware.py      # Rate limiting, security
├── Dockerfile             # Production container
├── docker-compose.yml     # Multi-service setup
└── DEPLOY.md             # Deployment guide
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /health` | Health check |
| `GET /metrics` | Request metrics |
| `POST /api/v1/analyze` | Analyze document |
| `POST /api/v1/query` | RAG query |
| `POST /api/v1/batch` | Batch analysis |
| `POST /api/v1/ingest` | Upload & index document |
| `GET /api/v1/status` | System status |

## Tech Stack

- **LLMs**: Claude 3, DeepSeek, GPT-4o-mini
- **Orchestration**: LangGraph, LangChain
- **Vector DB**: FAISS
- **API**: FastAPI + Uvicorn
- **Container**: Docker + Docker Compose

## Production Features

- Rate limiting (100 req/min)
- CORS configuration
- Security headers
- Health checks
- Request logging
- Error handling
- Docker + Docker Compose

---

Built with LangGraph + FastAPI + Docker

# Deployment Guide

## Quick Deploy Options

### Option 1: Railway (Recommended)

1. Push to GitHub
2. Go to railway.app
3. Connect your GitHub repo
4. Add environment variables:
   - `DEEPSEEK_API_KEY`
   - `ANTHROPIC_API_KEY`
   - `OPENAI_API_KEY`
   - `ENVIRONMENT=production`
5. Deploy! Get URL like: `https://legal-ai-api.up.railway.app`

### Option 2: Render

1. Create `render.yaml`:
```yaml
services:
  - type: web
    name: legal-ai-api
    env: docker
    dockerfilePath: ./Dockerfile
    envVars:
      - key: DEEPSEEK_API_KEY
        sync: false
      - key: ANTHROPIC_API_KEY
        sync: false
      - key: OPENAI_API_KEY
        sync: false
      - key: ENVIRONMENT
        value: production
```

2. Connect GitHub repo on Render, add env vars, deploy!

### Option 3: Google Cloud Run

```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/legal-ai-api
gcloud run deploy legal-ai-api \
    --image gcr.io/PROJECT_ID/legal-ai-api \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated
```

## Local Deployment

### Docker
```bash
cp .env.example .env
# Edit .env with your API keys
docker-compose up -d
curl http://localhost:8000/health
```

### Without Docker
```bash
pip install -r requirements-prod.txt
export DEEPSEEK_API_KEY=your_key
export ENVIRONMENT=production
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## API Testing

```bash
# Health check
curl http://localhost:8000/health

# Analyze document
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"document_text": "This is a contract...", "analysis_type": "contract_review"}'

# Query knowledge base
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the termination clause?", "k": 3}'
```

## Production Checklist

- [ ] Set `ENVIRONMENT=production`
- [ ] Set `DEBUG=false`
- [ ] Add all API keys
- [ ] Configure rate limiting
- [ ] Set up monitoring/logging
- [ ] Configure CORS origins
- [ ] Enable HTTPS

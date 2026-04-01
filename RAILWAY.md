# 🚂 Railway Deployment Guide

## Step 1: Sign Up for Railway

1. Go to [railway.app](https://railway.app)
2. Click **Sign Up**
3. Sign up with **GitHub** (easiest)
4. Verify your email

## Step 2: Create New Project

1. Click **"New Project"** button
2. Select **"Deploy from GitHub repo"**
3. Search for `legal_ai_assistant`
4. Click **Connect**

## Step 3: Add Environment Variables

Click on your deployed service → **Variables** tab

Add these variables:

```env
ENVIRONMENT=production
DEBUG=false
DEEPSEEK_API_KEY=sk-your-deepseek-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
OPENAI_API_KEY=sk-your-openai-key-here
```

## Step 4: Configure Deployment

Railway auto-detects Dockerfile. If not:

1. Go to **Settings** tab
2. Set:
   - **Build Command:** (leave empty - uses Dockerfile)
   - **Start Command:** `uvicorn api.main:app --host 0.0.0.0 --port 8000`

## Step 5: Deploy!

Click **Deploy** and wait ~2-3 minutes.

## Step 6: Get Your URL

After deployment:
- Click on **Settings** → **Networking** → **Public Networking**
- Enable public networking
- Your URL will be: `https://legal-ai-api.up.railway.app`

## Test Your API

```bash
# Health check
curl https://legal-ai-api.up.railway.app/health

# Analyze document
curl -X POST https://legal-ai-api.up.railway.app/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "document_text": "This is a software development contract between Company A and Company B...",
    "analysis_type": "contract_review"
  }'
```

## Update Your Profile README

Once deployed, update your README with the live URL:

```markdown
### [Legal AI Assistant](https://github.com/Sangamesh123hr/legal_ai_assistant)
> Production legal document analysis with multi-agent orchestration and RAG pipeline

- **Live API:** https://legal-ai-api.up.railway.app
```

---

## Troubleshooting

### Build Fails
- Check **Logs** tab for errors
- Common: Missing API keys → Add them in Variables

### Container Won't Start
- Health check failing → Check if `/health` endpoint works
- Port mismatch → Ensure `EXPOSE 8000` in Dockerfile

### Out of Memory
- Railway free tier: 512MB RAM
- Reduce batch processing size

---

## Free Tier Limits

| Resource | Limit |
|----------|-------|
| Projects | 5 |
| Total Spend | $5/month |
| Disk | 1GB |
| Network Out | 100GB/month |
| RAM | 512MB per service |

---

## Useful Railway Commands

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Link project
railway link

# View logs
railway logs

# Redeploy
railway redeploy
```

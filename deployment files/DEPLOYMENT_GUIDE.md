# Deployment Guide for Google Cloud Run

## Prerequisites
1. Google Cloud account with billing enabled
2. gcloud CLI installed and authenticated
3. Docker installed locally (for testing)

## Setup Steps

### 1. Set up Google Cloud Project
```bash
# Set your project ID
export PROJECT_ID=your-project-id
gcloud config set project $PROJECT_ID

# Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable cloudbuild.googleapis.com
```

### 2. Set Environment Variables
```bash
# Set your Gemini API key
export GEMINI_API_KEY=your-api-key

# Optional: Set GCS bucket for data storage
export GCS_BUCKET=your-bucket-name
```

### 3. Deploy Using Cloud Build (Recommended)
```bash
# Submit build to Cloud Build
gcloud builds submit --config cloudbuild.yaml

# Check deployment status
gcloud run services describe eda-dashboard --region=us-central1
```

### 4. Alternative: Manual Deployment
```bash
# Build Docker image
docker build -t gcr.io/$PROJECT_ID/eda-dashboard .

# Push to Container Registry
docker push gcr.io/$PROJECT_ID/eda-dashboard

# Deploy to Cloud Run
gcloud run deploy eda-dashboard \
  --image gcr.io/$PROJECT_ID/eda-dashboard \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 3600 \
  --set-env-vars GEMINI_API_KEY=$GEMINI_API_KEY
```

### 5. Test Locally (Optional)
```bash
# Build and run locally
docker build -t eda-dashboard .
docker run -p 8080:8080 \
  -e GEMINI_API_KEY=$GEMINI_API_KEY \
  eda-dashboard
```

## Configuration

### Memory and CPU Settings
- Default: 2 GB RAM, 2 vCPUs
- For larger datasets, increase to 4 GB RAM, 4 vCPUs
- Adjust timeout based on dataset size (default: 3600 seconds)

### Environment Variables
Set these in Cloud Run service settings:
- `GEMINI_API_KEY`: Your Gemini API key (required)
- `GCS_BUCKET`: GCS bucket for data storage (optional)
- `GCP_PROJECT_ID`: Your GCP project ID (optional)

## Monitoring and Logs

### View Logs
```bash
gcloud run services logs read eda-dashboard \
  --region us-central1 \
  --limit 50
```

### Monitor Performance
```bash
gcloud run services describe eda-dashboard \
  --region us-central1 \
  --format="value(status)"
```

## Cost Optimization
1. Use Cloud Run's autoscaling (min instances = 0)
2. Set appropriate memory limits
3. Use Cloud Build caching for faster builds
4. Monitor usage in Google Cloud Console

## Security Best Practices
1. Store API keys in Secret Manager
2. Use IAM roles for GCS access
3. Enable Cloud Armor for DDoS protection
4. Implement authentication for production

## Troubleshooting
- Check logs: `gcloud run services logs read eda-dashboard`
- Verify environment variables are set
- Ensure API keys are valid
- Check memory/CPU limits for large datasets

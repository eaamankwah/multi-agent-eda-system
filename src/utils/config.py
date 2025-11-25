class Config:
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', "Your_Gemini_api")
    MODEL_NAME = 'gemini-2.0-flash-exp'
    MAX_MISSING_THRESHOLD = 0.5
    OUTLIER_THRESHOLD = 3
    N_CLUSTERS_DEFAULT = 3
    CORRELATION_THRESHOLD = 0.5
    FIGURE_SIZE = (12, 6)
    COLOR_PALETTE = 'viridis'
    DASH_PORT = 8050

    # GCP Configuration (optional)
    GCP_PROJECT_ID = os.getenv('GCP_PROJECT_ID', 'your_google_project ID')
    GCS_BUCKET = os.getenv('GCS_BUCKET', 'your-bucket-name')

genai.configure(api_key=Config.GEMINI_API_KEY)
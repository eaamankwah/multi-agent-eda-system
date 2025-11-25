"""
Multi-Agent EDA System - Cloud Run Deployment
"""
import os
from main import MultiAgentEDASystem, Config
import pandas as pd

# Initialize system
eda_system = MultiAgentEDASystem()

# Load your data
# Option 1: From GCS
# from google.cloud import storage
# storage_client = storage.Client()
# bucket = storage_client.bucket(Config.GCS_BUCKET)
# blob = bucket.blob('data/your-data.csv')
# data = pd.read_csv(blob.open('rb'))

# Option 2: From local file (for testing)
data = pd.read_csv('sample_data.csv')

# Run analysis
results = eda_system.run_analysis(data)

# Get dashboard
app = eda_system.get_dashboard()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    if app:
        app.run_server(host='0.0.0.0', port=port, debug=False)
    else:
        print("Dashboard not available. Please install dash and dash-bootstrap-components.")

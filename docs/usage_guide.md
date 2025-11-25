# Usage Guide

- Run locally: `python src/app.py` and POST datasets to `/eda/run` via multipart form.
- Example: `curl -F "file=@data.csv" http://127.0.0.1:8000/eda/run`

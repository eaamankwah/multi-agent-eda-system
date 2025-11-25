from fastapi import FastAPI, UploadFile, File
import uvicorn
from orchestrator.controller import Orchestrator
import os, tempfile, pandas as pd

app = FastAPI(title="Multi-Agent EDA")

@app.post("/eda/run")
async def run_eda(file: UploadFile = File(...)):
    # Save incoming file temporarily
    tmpdir = tempfile.mkdtemp(prefix="eda_")
    fp = os.path.join(tmpdir, file.filename)
    with open(fp, "wb") as f:
        f.write(await file.read())
    # Basic: load using pandas
    try:
        df = pd.read_csv(fp)
    except Exception:
        df = pd.read_excel(fp)
    orch = Orchestrator()
    result = orch.run(df, run_name="api_run")
    return {"status": "completed", "artifacts": result}

if __name__ == "__main__":
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=False)

import pandas as pd
from orchestrator.controller import Orchestrator

def test_end_to_end(tmp_path):
    df = pd.DataFrame({
        "a": [1,2,3, None],
        "b": [4,5,None,6],
        "c": ["x","y","z", None]
    })
    orch = Orchestrator(artifact_root=str(tmp_path))
    res = orch.run(df, run_name="test")
    assert "report" in res

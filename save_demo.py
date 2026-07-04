import numpy as np, json
from ecg_triage import data
d = data.get_splits(); X, y = d["test"]; X = np.array(X); y = np.array(y)
for idx, name in [(315, "demo_MI.json"), (0, "demo_NORMAL.json")]:
    assert (idx == 315 and int(y[idx]) == 1) or (idx == 0 and int(y[idx]) == 0), "label mismatch"
    body = {"ecg_data": X[idx].astype(float).tolist(), "mode": "high_safety", "patient_id": f"PTBXL-{idx}"}
    open(name, "w").write(json.dumps(body))
    print("wrote", name, "true_label", int(y[idx]))

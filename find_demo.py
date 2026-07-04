import numpy as np, json, urllib.request
from ecg_triage import data
d = data.get_splits(); X, y = d["test"]; X = np.array(X); y = np.array(y)
def call(a):
    b = json.dumps({"ecg_data": np.array(a).astype(float).tolist(), "mode": "high_safety", "patient_id": "D"}).encode()
    r = urllib.request.Request("http://127.0.0.1:8000/predict", data=b, headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r).read())
found = {}
for i in range(len(X)):
    if len(found) >= 2: break
    res = call(X[i]); conf = res["confidence"]; p = res["mi_probability"]; lbl = int(y[i])
    if conf != "ABSTAINED":
        key = "MI" if lbl == 1 else "NORMAL"
        correct = (lbl == 1 and p >= 0.235) or (lbl == 0 and p < 0.235)
        if key not in found and correct:
            found[key] = i
            print(key, "idx", i, "true_label", lbl, "prob", round(p, 3), "risk", res["risk_category"], "conf", conf)
print("DONE", dict(found))

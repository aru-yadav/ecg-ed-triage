import numpy as np, json, urllib.request
from ecg_triage import data
d = data.get_splits(); X, y = d["test"]; X = np.array(X); y = np.array(y)
def call(a):
    b = json.dumps({"ecg_data": np.array(a).astype(float).tolist(), "mode":"high_safety","patient_id":"D"}).encode()
    r = urllib.request.Request("http://127.0.0.1:8000/predict", data=b, headers={"Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(r).read())
best = (-1, 0.0)
mi_idx = np.where(y == 1)[0][:60]   # scan first 60 true-MI records
for i in mi_idx:
    p = call(X[i])["mi_probability"]
    if p > best[1]: best = (int(i), p)
print("strongest MI so far: idx", best[0], "prob", round(best[1],3))

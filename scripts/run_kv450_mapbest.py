"""Bookkeeping between the stages of the KV450 run: writes runs/kv450_final/map_best.json (best spec-grid MAP from map_starts.json)
and runs/kv450_final/starts.json (6 pilot-chain start points spread in Omega_m from the 36 explore optima, map_explore.json)."""
import json
import numpy as np
import run_kv450_common as C

d = json.load(open(C.OUT / "map_starts.json"))
b = d["results"][0]
json.dump(dict(theta=b["theta"], names=d["names"], chi2=b["chi2"], lnpost=b["lnpost"], name=b["name"], z_grid="spec"),
          open(C.OUT / "map_best.json", "w"))
R = sorted(json.load(open(C.OUT / "map_explore.json"))["results"], key=lambda r: r["omega_m"])
pick = [R[i] for i in np.linspace(0, len(R) - 1, 6).astype(int)]
json.dump([r["theta"] for r in pick], open(C.OUT / "starts.json", "w"))
print("best MAP:", b["name"], b["chi2"], b["lnpost"], "| pilot starts:", [r["name"] for r in pick])

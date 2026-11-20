# One-off investigation script, not part of the Colab chain; see scripts/README.md.
"""Full-pair Bin6_Bin6 deviations of the stored full measurements from the exact (bin_slop 0, angle_slop 0) one, in OFFICIAL sigma.
Inputs (read-only): pilot_bin6/bin6_bin6_b0_a0_assembled.npz (exact), xi_chunked/assembled/ns/Bin6_Bin6.npz (0:default),
pilot_bin6/bin6_bin6_slop15_assembled.npz (1.5:default).  Output: angle_slop/fullpair_vs_exact.json."""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import dr5_angle_slop_study as S
PB = ROOT / "runs/dr5_measure/pilot_bin6"
L = lambda f: {k: np.asarray(v, float) for k, v in np.load(f).items()}
ex = L(PB / "bin6_bin6_b0_a0_assembled.npz")
ctx = S.official_ctx(6, 6); mf = ctx["mfac"]; r = ex["r_nom"]; g = ex["weight"]
out = {}
for nm, f in (("0:0.1", ROOT / "runs/dr5_measure/xi_chunked/assembled/ns/Bin6_Bin6.npz"), ("1.5:0.1", PB / "bin6_bin6_slop15_assembled.npz")):
    o = L(f); res = {}
    for k in ("xip", "xim"):
        d9 = S.rebin9(r, g, o[k]) - S.rebin9(r, g, ex[k])
        res[k + "9"] = (d9 / mf / ctx["sig9"][k]).tolist(); res["max_" + k + "9"] = float(np.abs(res[k + "9"]).max())
    Ed, Bd = S.cosebis_E(ctx, r, (o["xip"] - ex["xip"]) / mf, (o["xim"] - ex["xim"]) / mf)
    e = Ed / ctx["sE"]
    res["En"] = e.tolist(); res["max_En6"] = float(np.abs(e[:6]).max()); res["max_En20"] = float(np.abs(e).max())
    out[nm] = res
(ROOT / "runs/dr5_measure/angle_slop/fullpair_vs_exact.json").write_text(json.dumps(out, indent=1))
for k, v in out.items():
    print(k, {a: round(b, 4) for a, b in v.items() if a.startswith("max")})

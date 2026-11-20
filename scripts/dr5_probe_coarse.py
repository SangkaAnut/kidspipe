# One-off investigation script, not part of the Colab chain; see scripts/README.md.
"""Quick look at the coarse (20-bin) full-catalogue probe against the official xi+/-."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kidspipe.measure.official import PAIRS, load_official_xi

off = load_official_xi()
pi = PAIRS.index((0, 0))
d = np.load("/tmp/c_00_11.npz")
e = np.exp(np.linspace(np.log(2.0), np.log(300.0), 21))
o, xp, xm, th = off["npairs"][pi], off["xip"][pi], off["xim"][pi], off["r_nom"][pi]
rb = np.digitize(th, e) - 1
ox = np.array([np.sum(o[rb == k] * xp[rb == k]) / np.sum(o[rb == k]) for k in range(20)])
om = np.array([np.sum(o[rb == k] * xm[rb == k]) / np.sum(o[rb == k]) for k in range(20)])
onp = np.array([np.sum(o[rb == k]) for k in range(20)])

print("%-3s %12s %12s %8s %12s %12s %8s" % ("k", "xip_ours", "xip_off", "ratio",
                                            "xim_ours", "xim_off", "ratio"))
for k in range(20):
    print("%-3d %12.4e %12.4e %8.3f %12.4e %12.4e %8.3f"
          % (k + 1, d["xip"][k], ox[k], d["xip"][k] / ox[k], d["xim"][k], om[k], d["xim"][k] / om[k]))
print()
print("corr xip %.5f" % np.corrcoef(d["xip"], ox)[0, 1])
print("corr xim %.5f" % np.corrcoef(d["xim"], om)[0, 1])
print("median ratio xip %.4f  xim %.4f" % (np.median(d["xip"] / ox), np.median(d["xim"] / om)))
print("npairs ratio median %.5f" % np.median(d["npairs"] / onp))
print("npairs ratio spread %.2e" % np.std(d["npairs"] / onp))

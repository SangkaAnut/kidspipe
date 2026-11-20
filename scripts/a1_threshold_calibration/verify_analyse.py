import json, numpy as np
V="/mnt/d/cloning_kids_Kappa_pipeline/runs/dr5_measure/a1_salvage/threshold_calibration/standardised/verify/"
L=lambda c: json.load(open(V+f"e2ev2_{c}_ma2.json"))["stats"]
n,r,t10,t44=L("null"),L("real"),L("tilt10"),L("tilt44_20")
out={"n_null":len(n["minp"]),"n_real_seeds":len(r["minp"])}
for k in ("minp","fisher","stouffer","const"):
    nn=np.array(n[k]); rr=np.array(r[k])
    q={q_:float(np.percentile(nn,q_)) for q_ in (50,95,99)}
    thr95=q[95]
    out[k]={"null_quantiles":q,"null_mean":float(nn.mean()),"real_mean_over_table_seeds":float(rr.mean()),"real_sd_over_seeds":float(rr.std()),
      "real_e2e_pte":float((nn>=rr.mean()).mean()),"tilt10_power_at_null95":float((np.array(t10[k])>thr95).mean()),"tilt44_20_power_at_null95":float((np.array(t44[k])>thr95).mean()),
      "tilt10_median":float(np.median(t10[k])),"tilt44_median":float(np.median(t44[k]))}
    print(k,json.dumps(out[k]))
json.dump(out,open(V+"e2e_summary.json","w"),indent=1)

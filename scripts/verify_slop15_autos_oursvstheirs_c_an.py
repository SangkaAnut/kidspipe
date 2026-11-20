# One-off investigation script, not part of the Colab chain; see scripts/README.md.
import sys, numpy as np
d=np.load(sys.argv[1]); keys=sorted({k.split("__")[0] for k in d.files})
def g(k,q): return d[f"{k}__{q}"]
def stat(a,b,s):
    z=(g(a,'xip')-g(b,'xip'))/g(s,'sp'); zm=(g(a,'xim')-g(b,'xim'))/g(s,'sm')
    f=lambda z:"rms %.3f max %.2f lag1 %+.2f mean %+.4f"%(np.sqrt((z**2).mean()),abs(z).max(),np.corrcoef(z[:-1],z[1:])[0,1],z.mean())
    return f(z),f(zm)
for sl in ("1.5","0.0"):
    A,B,C,D,E=[f"{sl}_{x}" for x in ("A_f64","B_f32","C_f64_perm1","D_f32_perm1","E_f64_perm2")]
    print("== bin_slop",sl)
    for nm,a,b in [("f64 vs f32 (same order)  [= ours-vs-theirs analogue]",A,B),("perm1 vs orig (f64)",A,C),("perm1 vs perm2 (f64)",C,E),("perm1 vs orig (f32)",B,D),("f32perm1 vs f64orig",A,D)]:
        p,m=stat(a,b,A); print(f"{nm:55s} xi+ {p}\n{'':55s} xi- {m}")
    print("npairs rel diff f64 vs f32:",np.abs(g(A,'npairs')-g(B,'npairs')).sum()/g(A,'npairs').sum(),"  perm:",np.abs(g(A,'npairs')-g(C,'npairs')).sum()/g(A,'npairs').sum(), " nbins npairs differ f32:",int((g(A,'npairs')!=g(B,'npairs')).sum()),"perm:",int((g(A,'npairs')!=g(C,'npairs')).sum()))
# approximation error 1.5 vs 0
for nm,a,b in [("slop1.5 vs slop0 (f64)","1.5_A_f64","0.0_A_f64"),("slop1.5 f32 vs slop0 f64","1.5_B_f32","0.0_A_f64")]:
    p,m=stat(a,b,"0.0_A_f64"); print(f"{nm:35s} xi+ {p}\n{'':35s} xi- {m}")
print("total npairs 1.5/0:",g("1.5_A_f64",'npairs').sum()/g("0.0_A_f64",'npairs').sum())
# 9-bin
E9=np.exp(np.linspace(np.log(2),np.log(300),10)); r=g("0.0_A_f64","r")
def r9(a,b,q,s):
    out=[]
    for c in range(9):
        m=(r>=E9[c])&(r<E9[c+1]+(0.01 if c==8 else 0)); w=g(b,'weight')[m]
        out.append(((g(a,q)[m]-g(b,q)[m])*w).sum()/w.sum())
    return np.array(out)
# 9-bin sigma proxy: sqrt(sum w^2 sig^2)/sum w
def s9(k,q):
    o=[]
    for c in range(9):
        m=(r>=E9[c])&(r<E9[c+1]+(0.01 if c==8 else 0)); w=g(k,'weight')[m]; o.append(np.sqrt(((w*g(k,q)[m])**2).sum())/w.sum())
    return np.array(o)
for nm,a,b in [("f32-f64","1.5_B_f32","1.5_A_f64"),("perm-orig","1.5_C_f64_perm1","1.5_A_f64"),("1.5-0","1.5_A_f64","0.0_A_f64")]:
    for q,sq in (("xip","sp"),("xim","sm")):
        print("9bin",nm,q,"d/(shape-noise sigma, 200k)",np.round(r9(a,b,q,sq)/s9("0.0_A_f64",sq),4))

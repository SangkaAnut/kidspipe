import mpmath as mp
mp.mp.dps=60
def build(thmin,thmax,nmax):
    D=mp.log(mp.mpf(thmax)/thmin)
    def mom(k,a): return mp.quad(lambda z: z**k*mp.exp(a*z),[0,D])
    M={}
    def m(k,a):
        if (k,a) not in M: M[(k,a)]=mom(k,a)
        return M[(k,a)]
    polys=[]
    out=[]
    for n in range(1,nmax+1):
        deg=n+1
        rows=[[m(k,2) for k in range(deg+1)],[m(k,4) for k in range(deg+1)]]
        for p in polys:
            rows.append([sum(p[j]*m(j+k,1) for j in range(len(p))) for k in range(deg+1)])
        A=mp.matrix(rows)
        # null space: fix leading coeff=1, solve rest
        Asub=A[:, :deg]; b=-A[:,deg]
        c=mp.lu_solve(Asub,b)
        coef=[c[i] for i in range(deg)]+[mp.mpf(1)]
        nrm=sum(coef[i]*coef[j]*m(i+j,1) for i in range(deg+1) for j in range(deg+1))
        N=mp.sqrt((mp.exp(D)-1)/nrm)
        coef=[x*N for x in coef]
        polys.append(coef)
        roots=mp.polyroots(coef[::-1],maxsteps=500,extraprec=200)
        out.append((N,sorted([mp.re(r) for r in roots])))
    return out
if __name__=="__main__":
    for N,r in build(0.5,300,4):
        print(mp.nstr(N,10),[mp.nstr(x,8) for x in r])

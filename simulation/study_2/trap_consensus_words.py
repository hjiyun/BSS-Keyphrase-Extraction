"""consensus 트랩에서 acMH vs AWSGLD가 실제로 뽑은 단어(top-k)를 출력."""
import sys, numpy as np
sys.path.insert(0,"/home/jiyoon/BSS-Keyphrase-Extraction/code_JOC")
import keyphrase_functions_awsgld as M
import importlib.util
spec=importlib.util.spec_from_file_location("exp","acmh_vs_awsgld_4to10.py"); exp=importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)
from numpy.linalg import solve
from scipy.special import logsumexp

doc="2098"; g=exp.build_graph(doc); n=g['n']; d=0.85; ae=0.5
words=g['words']; G=solve(g['D'],g['A']); B=np.eye(n)-d*G.T; BtB=B.T@B
kw=sorted(g['truth']); TRUTH=set(kw); nkw=len(kw); K=5; sig2=2.0
nonkw=[i for i in range(n) if i not in TRUTH]
decoys=[nonkw[i::K] for i in range(K)]
decoy_of={}
for m,dc in enumerate(decoys):
    for i in dc: decoy_of[i]=m
COMMON,DECOY,TAU=2.0,4.0,200
centers=[]
for dc in decoys:
    u=np.full(n,-0.5); u[kw]=COMMON; u[dc]=DECOY; centers.append(u)
def lik(th): return -np.sum(np.log(np.clip(1-(1-ae)*M.inv_logit(th),1e-10,None)))
def Uk(th,u): C=(B@(th-u))@(B@(th-u)); return lik(th)+C/(2*sig2)
def Umix(th): return -logsumexp([-Uk(th,u) for u in centers])
def gUk(th,u):
    pi=np.clip(M.inv_logit(th),1e-10,1-1e-10); dpi=pi*(1-pi)
    return -(-(1-ae)*dpi/np.clip(1-(1-ae)*pi,1e-10,None))+BtB@(th-u)/sig2
def gUmix(th):
    Us=np.array([-Uk(th,u) for u in centers]); w=np.exp(Us-logsumexp(Us))
    return sum(w[i]*gUk(th,centers[i]) for i in range(K))
def mode(th): return int(np.argmin([Uk(th,u) for u in centers]))
T,BURN=50000,5000; ini=centers[0].copy()
def acmh(seed,step=0.3):
    np.random.seed(seed); th=ini.copy(); ths=np.zeros((T,n)); Uc=Umix(th)
    for t in range(T):
        star=th+np.random.randn(n)*step; Us=Umix(star)
        if np.log(np.random.rand()+1e-300)<(-Us+Uc): th=star;Uc=Us
        ths[t]=th
    return M.inv_logit(ths[BURN:]).mean(0)
def awsgld(seed,ZETA=10,eps0=0.5,Mr=500):
    np.random.seed(seed); th=ini.copy(); ths=np.zeros((T,n))
    aw=np.arange(1,Mr+1,dtype=float)/Mr;warm=300;emin=None;du=None;J=Mr-1;es=[]
    for t in range(T):
        epsk=eps0/((t+1)**0.5+10); Ut=Umix(th)/TAU; gU=gUmix(th)/TAU
        if t<warm:
            es.append(Ut)
            if t==warm-1: emn,emx=min(es),max(es);er=max(emx-emn,1);emin=emn-0.5*er;du=max((emx+0.5*er-emin)/Mr,1e-8)
            gm=1.0
        else:
            J=int(np.clip((Ut-emin)/du+1,1,Mr-1)); gm=np.clip(1+(ZETA/du)*(np.log(aw[J]+1e-12)-np.log(aw[J-1]+1e-12)),0.05,20)
        th=th-epsk*gm*gU+np.sqrt(2*epsk)*np.random.randn(n); th=np.clip(th,-700,700)
        if t>=warm:
            dec=min(1,100/((t+1)**0.75+1000));cw=aw[J];aw[J:]=aw[J:]+dec*cw*(1-aw[J:]);aw[:J]=aw[:J]-dec*cw*aw[:J];aw=np.clip(aw,1e-10,1)
        ths[t]=th
    return M.inv_logit(ths[BURN:]).mean(0)

def tag(i):
    if i in TRUTH: return "✅키워드"
    return f"❌미끼(모드{decoy_of[i]})" if i in decoy_of else "❌비키워드"

print(f"doc{doc}: 단어 {n}개, 진짜 키워드 {nkw}개")
print(f"진짜 키워드: {', '.join(words[i] for i in kw)}\n")
pa=acmh(1); pw=awsgld(1)
for name,pi in [("acMH (모드0 갇힘)",pa),("AWSGLD (모드 탈출)",pw)]:
    order=np.argsort(-pi)[:15]
    print(f"=== {name} top-15 ===")
    print(f"{'순위':>3} {'단어':<18} {'π':>6}  판정")
    hit=0
    for r,i in enumerate(order,1):
        if i in TRUTH: hit+=1
        print(f"{r:>3} {words[i]:<18} {pi[i]:>6.3f}  {tag(i)}")
    print(f"  top-15 중 진짜 키워드: {hit}/15\n")

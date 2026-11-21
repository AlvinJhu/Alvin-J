import sys, os; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); os.chdir(HERE)
"""大跌事件回顧：每次單日跌超過 4% 前，v3.1 模型（walk-forward 樣本外）的大跌機率、外資選擇權偏向、建議持股；並比較「警訊持續」規則。"""
import numpy as np, pandas as pd, risk_model as RM, model as M, futures_eval as FE
df=M.load(); X,yv,ydd,ew,lr=RM.build(df)
X["opt_fgn_bias"]=FE.fut_features(df,FE.load_futures(df.index))["opt_fgn_bias"]
X=X.dropna(); yv,ydd=yv.reindex(X.index),ydd.reindex(X.index)
har=["rv1","rv5","rv22"]; vc=har+["rv66","park5","neg5","vix"]
cc=[c for c in X if c!="opt_fgn_bias"]+["opt_fgn_bias"]
vf=np.exp(RM.wf(X[vc],yv,RM.ols)); pl=RM.wf(X[cc],ydd,RM.logit); px=RM.wf(X[cc],ydd,RM.xgb); p=(pl+px)/2
# 最後一段（2026）用到最新資料：walk-forward 已涵蓋
w=np.minimum(1,0.22/vf)*np.where(p>0.2,0.25,1)
r=lr.reindex(X.index)
big=r[r<-0.04].sort_values()
ev=[]; 
for d in big.index:
    if all(abs((d-e).days)>40 for e in ev): ev.append(d)
ev=sorted(ev)
rows=[]
for d in ev:
    i=X.index.get_loc(d)
    if i<25 or pd.isna(p.iloc[i-1]): continue
    peak=df.close.loc[:d].iloc[-60:].idxmax(); dd=df.close.loc[d:].iloc[:20].min()/df.close.loc[peak]-1
    rows.append({"大跌日":d.date(),"當日跌幅":round(np.expm1(r[d])*100,1),"自60日高點最大回落%":round(dd*100,1),
      "p_前20日":round(p.iloc[i-20],2),"p_前10日":round(p.iloc[i-10],2),"p_前5日":round(p.iloc[i-5],2),"p_前1日":round(p.iloc[i-1],2),
      "opt_前5日":round(X.opt_fgn_bias.iloc[i-5],3),"opt_前1日":round(X.opt_fgn_bias.iloc[i-1],3),
      "持股_前5日":round(w.iloc[i-5],2),"持股_前1日":round(w.iloc[i-1],2)})
t=pd.DataFrame(rows); print(t.to_string(index=False))
print("p 分位：", p.quantile([.5,.8,.9]).round(3).to_dict())
print(pd.DataFrame({"p":p,"opt":X.opt_fgn_bias,"vol":vf,"w":w}).loc["2026-10-20":].round(3).to_string())
t.to_csv("reports/crash_case_study.csv",index=False,encoding="utf-8-sig")

# 警訊持續規則：p 超過門檻後，減碼至少維持 N 日（不因單日機率回落就加回）
rr=(df.close.shift(-1)/df.close-1).reindex(X.index)
ok=p.notna()&vf.notna()&rr.notna(); P,V,RR=p[ok],vf[ok],rr[ok]
def run(thr,hold,cut=0.25,tg=0.22,band=0.2):
    flag=(P>thr).astype(int).values; act=np.zeros(len(flag)); last=-10**9
    for i,f in enumerate(flag):
        if f: last=i
        act[i]=1 if i-last<hold else 0
    raw=np.minimum(1,tg/V.values)*np.where(act==1,cut,1)
    w=np.empty(len(raw)); cur=raw[0]
    for i,x in enumerate(raw):
        if abs(x-cur)>band: cur=x
        w[i]=cur
    w=pd.Series(w,index=P.index); return RM.strat(w,RR), w
SPL=pd.Timestamp("2023-01-01"); res=[]
for thr in (0.2,0.25,0.3):
    for hold in (1,5,10,20):
        s,w=run(thr,hold)
        for per,m in (("2020-22",s.index<SPL),("2023-26",s.index>=SPL)):
            st=M.stats(s[m]); res.append({"thr":thr,"hold":hold,"期間":per,"Sharpe":round(st["Sharpe"],2),"MDD":round(st["最大回撤"],3),"CAGR":round(st["CAGR"],3),"持股":round(w[m].mean(),2)})
for per,m in (("2020-22",RR.index<SPL),("2023-26",RR.index>=SPL)):
    st=M.stats(RR[m]); res.append({"thr":"B&H","hold":"","期間":per,"Sharpe":round(st["Sharpe"],2),"MDD":round(st["最大回撤"],3),"CAGR":round(st["CAGR"],3),"持股":1})
R=pd.DataFrame(res); print(R.pivot_table(index=["thr","hold"],columns="期間",values=["Sharpe","MDD","CAGR"],aggfunc="first").to_string())
s,w=run(0.25,20); print(w.loc["2026-10-15":].round(2).to_string())
print("== 2023-01 ~ 2026-10（排除 11 月）")
for thr,hold in ((0.25,1),(0.25,10)):
    s,w=run(thr,hold); m=(s.index>=SPL)&(s.index<"2026-11-01"); st=M.stats(s[m]); print(thr,hold,{k:round(v,3) for k,v in st.items()})
m=(RR.index>=SPL)&(RR.index<"2026-11-01"); print("B&H",{k:round(v,3) for k,v in M.stats(RR[m]).items()})

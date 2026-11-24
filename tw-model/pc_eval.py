import sys, os; HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); os.chdir(HERE)
"""台指選擇權每日買賣權 P/C 比（成交量、未平倉）對 20 日內跌 7% 的預警力；與 v3.1 大跌模型同樣本比較。資料：fetch_local.py --optdaily。"""
import numpy as np, pandas as pd, model as M, risk_model as RM, futures_eval as FE
df=M.load(); X,yv,ydd,ew,lr=RM.build(df)
X["opt_fgn_bias"]=FE.fut_features(df,FE.load_futures(df.index))["opt_fgn_bias"]
o=pd.read_csv("data/ext/opt_txo_daily_agg.csv",parse_dates=["date"]).set_index("date").reindex(df.index)
P=pd.DataFrame(index=df.index)
P["pc_vol"]=np.log(o.vol_P/o.vol_C); P["pc_vol5"]=P.pc_vol.rolling(5).mean()
P["pc_oi"]=np.log(o.oi_P/o.oi_C); P["pc_oi_chg5"]=P.pc_oi.diff(5)
P["pc_oi_z60"]=(P.pc_oi-P.pc_oi.rolling(60).mean())/P.pc_oi.rolling(60).std()
P=P.shift(1)
A=pd.concat([X,P],axis=1).dropna(); y=ydd.reindex(A.index); SPL=RM.SPLIT
print("樣本",A.index[0].date(),A.index[-1].date(),len(A))
for c in P.columns:
    p=RM.wf(A[[c]],y,RM.logit); out=[]
    for per,m in (("18-22",A.index<SPL),("23-26",A.index>=SPL),("all",A.index==A.index)):
        s=M.score("dd","l",p[m],y[m],20); out.append(f"{per} {s['AUC']:.3f}[{s['AUC_95%下界']:.2f},{s['AUC_95%上界']:.2f}]")
    print(c," | ".join(out))
base=[c for c in X.columns]
res={}
for name,cols in (("v3.1",base),("v3.1+PC",base+list(P.columns))):
    p=(RM.wf(A[cols],y,RM.logit)+RM.wf(A[cols],y,RM.xgb))/2; res[name]=p; out=[]
    for per,m in (("18-22",A.index<SPL),("23-26",A.index>=SPL),("all",A.index==A.index)):
        s=M.score("dd","e",p[m],y[m],20); out.append(f"{per} {s['AUC']:.3f}[{s['AUC_95%下界']:.2f},{s['AUC_95%上界']:.2f}]")
    print(name," | ".join(out))
q=pd.qcut(A.pc_oi,5); print(pd.DataFrame({"q":q,"y":y}).groupby("q",observed=True).y.agg(["mean","size"]).round(3))
print(pd.concat([P.loc["2026-10-20":],res["v3.1+PC"].loc["2026-10-20":].rename("p_pc"),res["v3.1"].loc["2026-10-20":].rename("p")],axis=1).round(3).to_string())

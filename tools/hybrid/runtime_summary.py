import os, json, pandas as pd
rows=[]
for root,_,files in os.walk("results/eval"):
    if "metrics.json" in files:
        m=json.load(open(os.path.join(root,"metrics.json")))
        rows.append({"bench":m["bench"],"tag":m["tag"],"runtime_sec":m.get("runtime_sec",0.0)})
df=pd.DataFrame(rows)
if df.empty: print("no metrics found"); exit(0)
print(df.groupby("bench")["runtime_sec"].sum().div(3600).rename("hours").round(2).to_string())
print("\nTotal ChampSim eval hours:", round(df["runtime_sec"].sum()/3600,2))

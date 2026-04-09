from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "code"))
ART = ROOT / "artifacts_runtime" / "paper16_budgeted"
ART.mkdir(parents=True, exist_ok=True)
src = json.loads((ROOT / "data" / "reference_inputs" / "paper10_controller_summary.json").read_text(encoding="utf-8"))
summary={}
for system in ["heuristic","reranked"]:
    summary[system]={}
    for lam in [0.02,0.04,0.08,0.12,0.16,0.20]:
        summary[system][str(lam)]={}
        for name,row in src[system]["controllers"].items():
            summary[system][str(lam)][name]={"intent":row["intent"],"actions":row["actions"],"utility":row["intent"]-lam*row["actions"]}
sns.set_theme(style="whitegrid")
for kind,fn in [("utility","figure1_utility_lambda.png"),("intent","figure2_intent.png"),("actions","figure3_actions.png")]:
    fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=(kind!="actions"))
    for ax,system in zip(axes,["heuristic","reranked"]):
        for name,color in [("rank_only","#1f77b4"),("clarify_only","#ff7f0e"),("repair_only","#2ca02c"),("memory_clarify","#9467bd"),("full_controller","#d62728")]:
            xs=[0.02,0.04,0.08,0.12,0.16,0.20]
            ys=[summary[system][str(x)][name][kind] for x in xs]
            ax.plot(xs,ys,marker="o",label=name,color=color)
        ax.set_title(system); ax.set_xlabel("lambda")
        ax.set_ylabel(kind)
    axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(ART/fn,dpi=220); plt.close(fig)
(ART/"paper16_budgeted_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
print("Generated paper16 artifacts in",ART)

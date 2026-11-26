"""End-to-end pipeline:  python src/run_pipeline.py [--data path/to/your_cleaned.csv] [--colmap '{"job_title":"title",...}']
Without --data a synthetic multi-source dataset is generated (pipeline test only)."""
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")
from config import *
import collect, clean, extract, kg, forecast, explain, recommend

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})

DEMO_USERS = {"Final-year CSE student (Python, SQL, ML basics)": ["Python", "SQL", "Machine Learning", "Pandas", "Git"],
              "Web developer (JS, React, jQuery, PHP)": ["JavaScript", "React", "jQuery", "PHP", "Git"],
              "Data analyst (Excel, SQL, Tableau)": ["Excel", "SQL", "Tableau", "Communication"]}

def write_recommendations(rec, share):
    """Run demo personas (skills not present in the dataset are dropped) and write JSON + Power BI CSVs."""
    users = {k: [x for x in v if x in rec.sidx] for k, v in DEMO_USERS.items()}
    users = {k: v for k, v in users.items() if len(v) >= 2}
    if not users:                                   # dataset has none of the demo skills -> persona from its own top skills
        top = list(share.mean().sort_values(ascending=False).index[:4])
        users = {"Learner with the 4 most in-demand skills in the data": top}
    out_rec = {}
    for name, sk in users.items():
        r = rec.recommend(sk)
        out_rec[name] = dict(skills=r["user_skills"], roles=r["role_ranking"].round(3).to_dict("records"), target=r["target_role"],
                             gaps=r["skill_gaps"], companies=r["companies"], explanations=r["explanations"])
    json.dump(out_rec, open(OUT / "recommendations.json", "w"), indent=2, default=str)
    pd.DataFrame([dict(user=k, **{"rank": i + 1, "role": r["role"], "score": r["score"]}) for k, v in out_rec.items() for i, r in enumerate(v["roles"])]).to_csv(PBI / "recommendations_roles.csv", index=False)
    pd.DataFrame([dict(user=k, **g) for k, v in out_rec.items() for g in v["gaps"]]).to_csv(PBI / "recommendations_skills.csv", index=False)
    return out_rec

def figures_descriptive(share, H, gfeat):
    """Charts that need no forecast: demand over time, top skills, knowledge graph coloured by current demand."""
    mean = share.mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4.4)); t = mean.head(15)[::-1]
    ax.barh(t.index, t.values, color="#2980b9"); ax.set_xlabel("average demand (postings mentioning skill per 1000 postings)"); ax.set_title("Most demanded skills")
    plt.tight_layout(); plt.savefig(FIG / "fig6_skill_outlook.png"); plt.close()
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for sname in mean.index[:6]: ax.plot(share.index, share[sname].rolling(2, min_periods=1).mean(), label=sname)
    ax.set_title("Demand over time (top skills)"); ax.set_ylabel("postings per 1000"); ax.legend(fontsize=7, ncol=2)
    plt.tight_layout(); plt.savefig(FIG / "fig2_skill_trends.png"); plt.close()
    top = [s for s in gfeat.sort_values("pagerank", ascending=False).index[:28]]; sub = H.subgraph(top)
    if sub.number_of_nodes() < 2: return
    pos = nx.spring_layout(sub, seed=3, k=0.9, weight="weight")
    fig, ax = plt.subplots(figsize=(7.5, 6))
    nx.draw_networkx_edges(sub, pos, alpha=0.15, width=[0.3 + 1.5 * d["weight"] for *_, d in sub.edges(data=True)], ax=ax)
    nc = nx.draw_networkx_nodes(sub, pos, node_color=[mean.get(n, 0) for n in sub], cmap="YlGnBu", node_size=[200 + 4000 * gfeat.loc[n, "pagerank"] for n in sub], ax=ax, edgecolors="k", linewidths=0.4)
    nx.draw_networkx_labels(sub, pos, font_size=7, ax=ax); plt.colorbar(nc, label="average demand (per 1000 postings)", shrink=0.7); ax.axis("off"); ax.set_title("Skill knowledge graph (size = PageRank, colour = demand)")
    plt.tight_layout(); plt.savefig(FIG / "fig4_knowledge_graph.png"); plt.close()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data"); ap.add_argument("--colmap", default=None); ap.add_argument("--source", default="linkedin")
    a = ap.parse_args()
    # ---- Phase 1-2
    if a.data:
        raw = collect.load_real(a.data, a.source, json.loads(a.colmap) if a.colmap else None); mode = "REAL"
    else:
        raw = collect.generate_synthetic(); raw.to_csv(RAW / "jobs_raw_synthetic.csv", index=False); mode = "SYNTHETIC"
    print(f"[data] mode={mode}, rows={len(raw)}, sources={raw.source.value_counts().to_dict()}")
    df = clean.clean_jobs(raw); df.to_csv(PROC / "jobs_clean.csv", index=False)
    # ---- Phase 3
    df = extract.extract_all(df)
    cnt, share, tot = extract.monthly_skill_counts(df)
    keep = cnt.columns[cnt.sum() >= MIN_SKILL_POSTINGS]; cnt, share = cnt[keep], share[keep]
    share.to_csv(PROC / "monthly_share.csv"); cnt.to_csv(PROC / "monthly_counts.csv")
    emb = extract.skill_embeddings()
    # ---- Phase 4
    G = kg.build_graph(df); H = kg.skill_graph(G)
    skills = list(share.columns); gfeat = kg.graph_features(H, skills); A = kg.neighbor_matrix(H, skills)
    kg.export_neo4j(G, NEO)

    # Skip forecasting and dependent phases when history is insufficient
    min_history = max(16, LOOKBACK + HORIZON + TEST_MONTHS)

    if len(share) < min_history:
        print(f"[forecast] SKIPPED: {len(share)} months available; "
              f"at least {min_history} months required.")

        cnt.reset_index(names="month").melt(
            "month", var_name="skill", value_name="postings"
        ).to_csv(PBI / "fact_skill_monthly.csv", index=False)

        share.reset_index(names="month").melt(
            "month", var_name="skill", value_name="share_per_1000"
        ).to_csv(PBI / "fact_skill_share.csv", index=False)

        df.drop(columns=["description"]).assign(
            skills=lambda d: d.skills.map("|".join)
        ).to_csv(PBI / "fact_postings.csv", index=False)

        ed = pd.DataFrame([
            dict(src=u[1], dst=v[1], weight=d["weight"])
            for u, v, d in G.edges(data=True)
            if d["rel"] == "CO_OCCURS"
        ])
        ed.to_csv(PBI / "skill_graph_edges.csv", index=False)

        # Recommendations still work without a forecast: use CURRENT demand instead of forecast growth
        fc0 = pd.DataFrame({"skill": share.columns, "current_level": share.mean().values})
        rec = recommend.CareerRecommender(G, H, fc0, demand_only=True)
        write_recommendations(rec, share)
        figures_descriptive(share, H, gfeat)
        print("[recommend] done in current-demand mode (no forecast)")

        summary = dict(
            mode=mode,
            rows=int(len(df)),
            months=int(len(share)),
            skills=len(skills),
            kg_nodes=G.number_of_nodes(),
            kg_edges=G.number_of_edges(),
            forecast_status="SKIPPED",
            forecast_message=f"Only {len(share)} months available; "
                             f"{min_history} required.",
            xai_status="SKIPPED",
            recommendations_status="OK (current-demand mode, no forecast)",
            best_model=None
        )

        (OUT / "run_summary.json").write_text(
            json.dumps(summary, indent=2)
        )

        print("[done] Skill extraction, knowledge graph and recommendations completed.")
        print("[done] Outputs saved to", OUT)
        return
    # ---- Phase 5 + 6
    results, ctx = forecast.fit_predict_all(share, A, gfeat)
    results.to_csv(OUT / "model_comparison.csv"); print("\n", results.to_string(), "\n")
    fc, final_model = forecast.future_forecast(share, A, gfeat, results)
    fc["pagerank"] = gfeat.loc[fc.skill, "pagerank"].values; fc["community"] = gfeat.loc[fc.skill, "community"].values
    fc["category"] = [collect.SKILLS[s][0] for s in fc.skill]
    fc.to_csv(OUT / "skill_forecast.csv", index=False)
    # ---- Phase 7
    hg = ctx["hg_model"]

    # If PyTorch HGTSP was used, train a separate explainable surrogate.
    # This surrogate explains its own predictions, not the PyTorch model directly.
    if hg is None:
        from sklearn.ensemble import HistGradientBoostingRegressor
        hg = HistGradientBoostingRegressor(
            max_iter=500,
            learning_rate=0.04,
            max_depth=5,
            l2_regularization=1.0,
            random_state=SEED
        ).fit(ctx["Xtr"], ctx["tr"].yn)

        xai_model_type = "HistGradientBoosting surrogate for HGTSP-PyTorch"
        print("[xai] Using separate HistGradientBoosting surrogate for explanations.")
    else:
        xai_model_type = "HGTSP-lite (HistGradientBoostingRegressor)"

    if hg is not None:
        gi = explain.global_importance(hg, ctx["Xte"], ctx["te"].yn.values, ctx["names"]); gi.to_csv(OUT / "global_importance.csv", index=False)
        fam = gi.groupby("family").importance.sum().sort_values(ascending=False); fam.to_csv(OUT / "importance_by_family.csv")
        print("[xai] importance by family:\n", fam.round(4).to_string())
        sv = explain.shap_values(hg, ctx["Xtr"], ctx["Xte"], ctx["names"])
        local = {}
        for sk in list(fc[fc.reliable].skill[:3]) + list(fc[fc.reliable].skill[-2:]):
            if sk in skills: local[sk] = explain.explain_skill(hg, ctx, skills, sk)
        json.dump({k: dict(pred=v["predicted_ratio"], top_features=v["local"]) for k, v in local.items()}, open(OUT / "local_explanations.json", "w"), indent=2)
    # ---- Phase 8
    rec = recommend.CareerRecommender(G, H, fc)
    out_rec = write_recommendations(rec, share)
    # ---- Power BI exports
    cnt.reset_index(names="month").melt("month", var_name="skill", value_name="postings").to_csv(PBI / "fact_skill_monthly.csv", index=False)
    share.reset_index(names="month").melt("month", var_name="skill", value_name="share_per_1000").to_csv(PBI / "fact_skill_share.csv", index=False)
    fc.to_csv(PBI / "dim_skill_forecast.csv", index=False)
    df.drop(columns=["description"]).assign(skills=lambda d: d.skills.map("|".join)).to_csv(PBI / "fact_postings.csv", index=False)
    results.reset_index(names="model").to_csv(PBI / "model_comparison.csv", index=False)
    if hg is not None: gi.to_csv(PBI / "xai_global_importance.csv", index=False)
    ed = pd.DataFrame([dict(src=u[1], dst=v[1], weight=d["weight"]) for u, v, d in G.edges(data=True) if d["rel"] == "CO_OCCURS"]); ed.to_csv(PBI / "skill_graph_edges.csv", index=False)
    figures(share, results, fc, G, H, gfeat, ctx, skills, df)
    (OUT / "run_summary.json").write_text(json.dumps(dict(
        mode=mode,
        rows=int(len(df)),
        months=int(len(share)),
        skills=len(skills),
        kg_nodes=G.number_of_nodes(),
        kg_edges=G.number_of_edges(),
        forecast_status="COMPLETED",
        xai_status="COMPLETED",
        xai_model=xai_model_type,
        recommendations_status="COMPLETED",
        best_model=results.MAE.idxmin()
    ), indent=2))
    print("\n[done] outputs ->", OUT)

def figures(share, results, fc, G, H, gfeat, ctx, skills, df):
    # 1 model comparison
    r = results.sort_values("MAE"); fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
    for a_, m in zip(ax, ["MAE", "RMSE", "MAPE"]):
        c = ["#c0392b" if ("HGTSP" in i) else "#7f8c8d" for i in r.index]; a_.barh(r.index, r[m], color=c); a_.set_title(m + (" (%)" if m == "MAPE" else "")); a_.invert_yaxis()
        if m != "MAE": a_.set_yticklabels([])
    plt.tight_layout(); plt.savefig(FIG / "fig1_model_comparison.png"); plt.close()
    # 2 trends of top emerging / declining
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for a_, sk, t in [(ax[0], list(fc[fc.reliable].skill.head(5)), "Predicted emerging skills"), (ax[1], list(fc[fc.reliable].sort_values("growth_pct").skill.head(5)), "Predicted declining skills")]:
        for s in sk: a_.plot(share.index, share[s].rolling(3, min_periods=1).mean(), label=s)
        a_.set_title(t); a_.set_ylabel("postings per 1000"); a_.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(FIG / "fig2_skill_trends.png"); plt.close()
    # 3 forecast vs actual on held-out months for two skills
    te = ctx["te"]; p = ctx["preds"][ctx["hg_name"]]; fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    for a_, sk in zip(ax, [fc[fc.reliable].skill.iloc[0], "Python", fc[fc.reliable].skill.iloc[-1]]):
        if sk not in skills: continue
        s = skills.index(sk); m = (te.s.values == s)
        a_.plot(share.index[-30:], share[sk].values[-30:], "k-", label="actual")
        a_.plot(share.index[te.tgt.values[m]], p[m], "r--o", ms=3, label=ctx["hg_name"].split(" ")[0]); a_.set_title(sk); a_.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(FIG / "fig3_forecast_vs_actual.png"); plt.close()
    # 4 knowledge graph
    top = [s for s in gfeat.sort_values("pagerank", ascending=False).index[:28]]; sub = H.subgraph(top)
    pos = nx.spring_layout(sub, seed=3, k=0.9, weight="weight"); gr = fc.set_index("skill").trend_pct_per_year.reindex(top).fillna(0).clip(-60, 100)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    nx.draw_networkx_edges(sub, pos, alpha=0.15, width=[0.3 + 1.5 * d["weight"] for *_, d in sub.edges(data=True)], ax=ax)
    nc = nx.draw_networkx_nodes(sub, pos, node_color=gr.values, cmap="RdYlGn", node_size=[200 + 4000 * gfeat.loc[n, "pagerank"] for n in sub], ax=ax, edgecolors="k", linewidths=0.4)
    nx.draw_networkx_labels(sub, pos, font_size=7, ax=ax); plt.colorbar(nc, label="24-month demand trend (% per year)", shrink=0.7); ax.axis("off"); ax.set_title("Skill knowledge graph (size = PageRank, colour = 24-month trend)")
    plt.tight_layout(); plt.savefig(FIG / "fig4_knowledge_graph.png"); plt.close()
    # 5 XAI
    p = OUT / "importance_by_family.csv"
    if p.exists():
        fam = pd.read_csv(p, index_col=0).iloc[:, 0]; gi = pd.read_csv(OUT / "global_importance.csv").head(10)
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.4)); ax[0].barh(fam.index[::-1], fam.values[::-1], color="#2980b9"); ax[0].set_title("Importance by feature family")
        ax[1].barh(gi.feature[::-1], gi.importance[::-1], color="#27ae60"); ax[1].set_title("Top-10 features (permutation, MAE increase)")
        plt.tight_layout(); plt.savefig(FIG / "fig5_explainability.png"); plt.close()
    # 6 top growth bars
    fcr = fc[fc.reliable].sort_values('growth_pct', ascending=False)
    t = pd.concat([fcr.head(10), fcr.tail(6)]); fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(t.skill[::-1], t.growth_pct[::-1], color=["#c0392b" if v < 0 else "#27ae60" for v in t.growth_pct[::-1]]); ax.set_xlabel("forecast change in demand, next 3 months (%)"); ax.set_title("Skill demand outlook")
    plt.tight_layout(); plt.savefig(FIG / "fig6_skill_outlook.png"); plt.close()

if __name__ == "__main__":
    main()

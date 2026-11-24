"""Phase 7 - Explainable AI: SHAP (if installed) + LIME (if installed) with self-contained fallbacks
(permutation importance for global, weighted local-linear surrogate = LIME algorithm for local)."""
import numpy as np, pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge

def family(name):
    if name.startswith("nbr_"): return "Graph neighbours' demand"
    if name.startswith("lag_"): return "Own recent demand (lags)"
    if name in ("month_sin", "month_cos"): return "Seasonality"
    if name in ("mom_3m", "trend_12m", "volatility"): return "Momentum / volatility"
    return "KG structure (PageRank, degree, clustering, community)"

def global_importance(model, X, y, names, n_repeats=8, seed=0):
    r = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=seed, scoring="neg_mean_absolute_error")
    df = pd.DataFrame({"feature": names, "importance": r.importances_mean, "std": r.importances_std})
    df["family"] = df.feature.map(family)
    return df.sort_values("importance", ascending=False).reset_index(drop=True)

def shap_values(model, Xtr, Xte, names, max_n=300):
    """Real SHAP when available. Returns (values, X) or None."""
    try:
        import shap
        bg = shap.sample(Xtr, 100, random_state=0)
        ex = shap.Explainer(model.predict, bg)
        sv = ex(Xte[:max_n])
        return sv.values, Xte[:max_n]
    except Exception as e:
        print("[xai] shap unavailable ->", type(e).__name__); return None

def lime_local(model, x, Xtr, names, n=2000, kernel_width=0.75, seed=0, topk=8):
    """Local explanation of one prediction. Uses `lime` if installed, else our own implementation of the
    same algorithm (perturb -> weight by proximity -> fit sparse linear surrogate)."""
    try:
        from lime.lime_tabular import LimeTabularExplainer
        ex = LimeTabularExplainer(Xtr, feature_names=names, mode="regression", random_state=seed)
        e = ex.explain_instance(x, model.predict, num_features=topk, num_samples=n)
        return [(a, b) for a, b in e.as_list()]
    except Exception:
        rng = np.random.default_rng(seed)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        keep = rng.random((n, len(x))) < 0.5                         # which features keep the instance value
        Z = np.where(keep, x, rng.normal(mu, sd, (n, len(x))))      # others are resampled from the data distribution
        w = np.exp(-(((Z - x) / sd) ** 2).mean(1) / kernel_width ** 2)
        y = model.predict(Z)
        sur = Ridge(alpha=1.0).fit((Z - mu) / sd, y, sample_weight=w)
        contrib = sur.coef_ * ((x - mu) / sd)                        # contribution of each feature for THIS instance
        idx = np.argsort(-np.abs(contrib))[:topk]
        return [(names[i], float(contrib[i])) for i in idx]

def family_contrib(local):
    d = {}
    for n, c in local:
        n = n.split(" ")[0] if " " in n and not n.startswith(("lag_", "nbr_")) else n
        d[family(n)] = d.get(family(n), 0) + c
    return d

def explain_skill(model, ctx, skill_names, skill, topk=6):
    te = ctx["te"]; names = ctx["names"]
    s = skill_names.index(skill)
    rows = np.where(te.s.values == s)[0]
    i = rows[-1]                                                     # latest test window for this skill
    x = ctx["Xte"][i]; pred = float(model.predict(x[None])[0])
    loc = lime_local(model, x, ctx["Xtr"], names, topk=topk)
    return dict(skill=skill, predicted_ratio=pred, local=loc)

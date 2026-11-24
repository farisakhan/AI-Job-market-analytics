"""Phase 5 - Temporal learning & benchmarking.

Compares: Naive, Seasonal-Naive, Random Forest, XGBoost (HistGB fallback), LSTM (torch), Prophet,
Temporal Fusion Transformer (see tft_model.py) and the proposed HGTSP (hgtsp_torch.py) or its
sklearn-based graph-augmented variant 'HGTSP-lite' when PyTorch is unavailable.
Metrics: MAE, RMSE, MAPE.
"""
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from config import LOOKBACK, HORIZON, TEST_MONTHS, SEED

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

def metrics(y, p, eps=1e-6):
    y, p = np.asarray(y, float), np.asarray(p, float)
    mae = np.mean(np.abs(y - p)); rmse = np.sqrt(np.mean((y - p) ** 2))
    mape = np.mean(np.abs(y - p) / np.maximum(np.abs(y), 1.0)) * 100   # denominators floored at 1 post/1000 (avoids blow-up near zero)
    return dict(MAE=mae, RMSE=rmse, MAPE=mape)

def make_windows(share, A=None, gfeat=None, L=LOOKBACK, H=HORIZON, test_months=TEST_MONTHS):
    """share: DataFrame months x skills. Returns dict with train/test feature blocks."""
    Y = share.values.T.astype(float)                 # S x T
    S, T = Y.shape
    nbr = (A @ Y) if A is not None else np.zeros_like(Y)   # graph message passing: neighbour-mean demand
    months = share.index
    recs = []
    for s in range(S):
        for t in range(L - 1, T - H):
            tgt = t + H
            w = Y[s, t - L + 1:t + 1]; wn = nbr[s, t - L + 1:t + 1]
            sc = w.mean() + 1e-3
            recs.append(dict(s=s, t=t, tgt=tgt, scale=sc, y=Y[s, tgt], yn=Y[s, tgt] / sc,
                             w=w / sc, wn=wn / (wn.mean() + 1e-3),
                             sea=Y[s, tgt - 12] / sc if tgt - 12 >= 0 else w[-1] / sc,
                             moy=months[tgt].month))
    d = pd.DataFrame(recs)
    split = T - test_months
    tr = d[d.tgt < split]; te = d[d.tgt >= split]
    return d, tr, te, split

def feat_matrix(d, gfeat=None, graph=True):
    W = np.stack(d.w.values); WN = np.stack(d.wn.values)
    mo = d.moy.values
    parts = [W, np.c_[np.sin(2 * np.pi * mo / 12), np.cos(2 * np.pi * mo / 12)],
             (W[:, -1] - W[:, -4])[:, None], (W[:, -1] - W[:, 0])[:, None], W.std(1)[:, None]]
    if graph:
        parts += [WN, (WN[:, -1] - WN[:, -4])[:, None]]
        if gfeat is not None:
            G = gfeat.values[d.s.values]
            parts.append(G)
    return np.hstack(parts)

def feat_names(L=LOOKBACK, graph=True, gcols=()):
    n = [f"lag_{L - 1 - i}" for i in range(L)] + ["month_sin", "month_cos", "mom_3m", "trend_12m", "volatility"]
    if graph: n += [f"nbr_lag_{L - 1 - i}" for i in range(L)] + ["nbr_mom_3m"] + list(gcols)
    return n

def fit_predict_all(share, A, gfeat, verbose=True):
    d, tr, te, split = make_windows(share, A, gfeat)
    res, preds = {}, {}
    ytest = te.y.values; sc = te.scale.values
    def record(name, yn_hat):
        p = np.maximum(yn_hat, 0) * sc
        res[name] = metrics(ytest, p); preds[name] = p
        if verbose: print(f"[forecast] {name:<28} MAE={res[name]['MAE']:.3f} RMSE={res[name]['RMSE']:.3f} MAPE={res[name]['MAPE']:.2f}%")
    record("Naive (last value)", np.stack(te.w.values)[:, -1])
    record("Seasonal Naive", te.sea.values)

    Xtr, Xte = feat_matrix(tr, None, False), feat_matrix(te, None, False)
    rf = RandomForestRegressor(300, min_samples_leaf=2, n_jobs=-1, random_state=SEED).fit(Xtr, tr.yn)
    record("Random Forest", rf.predict(Xte))
    if XGBRegressor is not None:
        xg = XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.9, random_state=SEED).fit(Xtr, tr.yn); nm = "XGBoost"
    else:
        xg = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_depth=4, random_state=SEED).fit(Xtr, tr.yn); nm = "XGBoost (HistGB fallback)"
    record(nm, xg.predict(Xte))

    # LSTM (optional)
    try:
        from lstm_model import fit_lstm
        record("LSTM", fit_lstm(np.stack(tr.w.values), tr.yn.values, np.stack(te.w.values)))
    except Exception as e:
        if verbose: print("[forecast] LSTM skipped:", type(e).__name__, "(install torch)")
    # Prophet (optional; per-skill, trained only on pre-split months)
    try:
        from prophet_model import prophet_predict
        pp = prophet_predict(share, split, HORIZON, te)
        res["Prophet"] = metrics(ytest, pp); preds["Prophet"] = pp
        if verbose: print(f"[forecast] Prophet MAE={res['Prophet']['MAE']:.3f}")
    except Exception as e:
        if verbose: print("[forecast] Prophet skipped:", type(e).__name__)

    # Proposed model
    Xtr_g, Xte_g = feat_matrix(tr, gfeat, True), feat_matrix(te, gfeat, True)
    try:
        from hgtsp_torch import fit_hgtsp
        record("HGTSP (proposed, torch)", fit_hgtsp(share, A, gfeat, tr, te, split))
        hg_model, hg_name = None, "HGTSP (proposed, torch)"
    except Exception as e:
        if verbose: print("[forecast] torch HGTSP skipped:", type(e).__name__, "-> using HGTSP-lite")
        hg_model = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.04, max_depth=5, l2_regularization=1.0,
                                                 random_state=SEED).fit(Xtr_g, tr.yn)
        record("HGTSP-lite (graph-augmented)", hg_model.predict(Xte_g)); hg_name = "HGTSP-lite (graph-augmented)"
    # ablation: same model without graph information
    ab = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.04, max_depth=5, l2_regularization=1.0, random_state=SEED).fit(Xtr, tr.yn)
    record("Ablation: no graph", ab.predict(Xte))
    ctx = dict(d=d, tr=tr, te=te, split=split, Xtr=Xtr_g, Xte=Xte_g, hg_model=hg_model, rf=rf, preds=preds,
               hg_name=hg_name, names=feat_names(graph=True, gcols=gfeat.columns))
    return pd.DataFrame(res).T.round(4), ctx

def future_forecast(share, A, gfeat, model_fit_df, L=LOOKBACK, H=HORIZON):
    """Train on ALL available windows and forecast demand H months beyond the last month."""
    d, _, _, _ = make_windows(share, A, gfeat, test_months=0)
    m = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.04, max_depth=5, l2_regularization=1.0, random_state=SEED)
    m.fit(feat_matrix(d, gfeat, True), d.yn)
    Y = share.values.T; S, T = Y.shape
    nbr = A @ Y
    last_month = share.index[-1]
    tgt_month = (last_month + pd.DateOffset(months=H)).month
    rows = []
    for s in range(S):
        w = Y[s, T - L:]; wn = nbr[s, T - L:]; sc = w.mean() + 1e-3
        rows.append(dict(s=s, w=w / sc, wn=wn / (wn.mean() + 1e-3), moy=tgt_month, scale=sc))
    q = pd.DataFrame(rows)
    yhat = np.maximum(m.predict(feat_matrix(q, gfeat, True)), 0) * q.scale.values
    now = Y[:, T - 6:].mean(1)                       # stable baseline: trailing 6-month mean
    out = pd.DataFrame({"skill": share.columns, "current_level": now, "forecast_level": yhat})
    out["growth_pct"] = (out.forecast_level - out.current_level) / np.maximum(out.current_level, 1) * 100
    out["volatility_cv"] = Y[:, T - 12:].std(1) / (Y[:, T - 12:].mean(1) + 1e-9)
    # descriptive trend: log-linear slope over the last 24 months, expressed as % change per year
    t24 = np.arange(24); Y24 = np.log1p(Y[:, T - 24:])
    out["trend_pct_per_year"] = (np.exp(np.polyfit(t24, Y24.T, 1)[0] * 12) - 1) * 100
    out["reliable"] = (out.current_level >= 20) & (out.volatility_cv < 0.4)   # >=2% of postings and not wildly noisy
    out.loc[~out.reliable, "growth_pct"] = out.loc[~out.reliable, "growth_pct"].clip(-25, 25)
    out = out.sort_values(["reliable", "growth_pct"], ascending=[False, False]).reset_index(drop=True)
    return out, m

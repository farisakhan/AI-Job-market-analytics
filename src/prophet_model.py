"""Prophet baseline: one model per skill trained on months < split, predicts each test target month."""
import numpy as np, pandas as pd
from prophet import Prophet

def prophet_predict(share, split, H, te):
    out = np.zeros(len(te)); te = te.reset_index(drop=True)
    for s in te.s.unique():
        df = pd.DataFrame({"ds": share.index[:split], "y": share.iloc[:split, s].values})
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False).fit(df)
        idx = te.index[te.s == s]
        fut = pd.DataFrame({"ds": share.index[te.loc[idx, "tgt"].values]})
        out[idx] = np.maximum(m.predict(fut)["yhat"].values, 0)
    return out

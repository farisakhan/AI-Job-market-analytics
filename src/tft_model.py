"""Temporal Fusion Transformer baseline using pytorch-forecasting (pip install pytorch-forecasting lightning).
Run separately:  python src/tft_model.py   (needs data/processed/monthly_share.csv produced by run_pipeline.py)."""
import pandas as pd, numpy as np
from config import PROC, LOOKBACK, HORIZON, TEST_MONTHS

def run_tft(max_epochs=30):
    import torch, lightning.pytorch as pl
    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.metrics import QuantileLoss
    share = pd.read_csv(PROC / "monthly_share.csv", index_col=0, parse_dates=True)
    long = share.reset_index(names="date").melt("date", var_name="skill", value_name="demand")
    long["time_idx"] = (long.date.dt.year - long.date.dt.year.min()) * 12 + long.date.dt.month
    long["time_idx"] -= long.time_idx.min(); long["month"] = long.date.dt.month.astype(str)
    cutoff = long.time_idx.max() - TEST_MONTHS
    tr = TimeSeriesDataSet(long[long.time_idx <= cutoff], time_idx="time_idx", target="demand", group_ids=["skill"],
        max_encoder_length=LOOKBACK, max_prediction_length=HORIZON, static_categoricals=["skill"],
        time_varying_known_categoricals=["month"], time_varying_unknown_reals=["demand"],
        target_normalizer=GroupNormalizer(groups=["skill"]), add_relative_time_idx=True)
    va = TimeSeriesDataSet.from_dataset(tr, long, predict=True, stop_randomization=True)
    tl = tr.to_dataloader(train=True, batch_size=128); vl = va.to_dataloader(train=False, batch_size=256)
    tft = TemporalFusionTransformer.from_dataset(tr, learning_rate=0.03, hidden_size=32, attention_head_size=4,
        dropout=0.1, hidden_continuous_size=16, loss=QuantileLoss())
    pl.Trainer(max_epochs=max_epochs, gradient_clip_val=0.1, accelerator="auto").fit(tft, tl, vl)
    pred = tft.predict(vl, mode="prediction", return_x=False)
    print("TFT done; use tft.interpret_output() for variable importance + attention.")
    return tft, pred

if __name__ == "__main__":
    run_tft()

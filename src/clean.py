"""Phase 2 - Data cleaning (Pandas)."""
import re, html
import pandas as pd, numpy as np

def clean_text(t):
    if not isinstance(t, str): return ""
    t = html.unescape(t)
    t = re.sub(r"<[^>]+>", " ", t)               # strip HTML
    t = re.sub(r"http\S+|www\.\S+", " ", t)      # urls
    t = re.sub(r"\s+", " ", t).strip()
    return t

def clean_jobs(df):
    df = df.copy()
    n0 = len(df)
    df["title"] = df["title"].fillna("").map(clean_text)
    df["company"] = df["company"].fillna("Unknown").map(lambda s: clean_text(s).title())
    df["location"] = df["location"].fillna("Unknown").map(clean_text)
    df["description"] = df["description"].map(clean_text)
    df["date_posted"] = pd.to_datetime(df["date_posted"], errors="coerce")
    df = df.dropna(subset=["date_posted"])
    df = df[df["description"].str.len() > 30]
    # near-duplicate removal across sources (same title/company/month/desc)
    df["month"] = df["date_posted"].dt.to_period("M").dt.to_timestamp()
    key = df["title"].str.lower() + "|" + df["company"].str.lower() + "|" + df["description"].str.lower().str[:120] + "|" + df["month"].astype(str)
    df = df.loc[~key.duplicated()]
    # normalise role titles: strip seniority prefixes
    df["role"] = df["title"].str.replace(r"^(senior|sr\.?|junior|jr\.?|lead|principal|associate)\s+", "", case=False, regex=True).str.title()
    df["salary"] = pd.to_numeric(df["salary"], errors="coerce")
    df.loc[(df.salary < 1e5) | (df.salary > 5e7), "salary"] = np.nan
    print(f"[clean] {n0} -> {len(df)} rows after cleaning")
    return df.reset_index(drop=True)

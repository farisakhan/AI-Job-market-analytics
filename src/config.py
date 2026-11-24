"""Central configuration for the HGTSP project."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
PBI = OUT / "powerbi"
NEO = OUT / "neo4j"
for p in (RAW, PROC, FIG, PBI, NEO):
    p.mkdir(parents=True, exist_ok=True)

SEED = 42
LOOKBACK = 12        # months of history fed to the model
HORIZON = 3          # forecast demand HORIZON months ahead
TEST_MONTHS = 6      # last N months held out for evaluation
MIN_SKILL_POSTINGS = 30

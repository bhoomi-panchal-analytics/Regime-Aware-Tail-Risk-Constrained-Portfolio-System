import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def safe_read_csv(filename):
    file_path = DATA_DIR / filename

    if not file_path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        return df
    except Exception:
        return pd.DataFrame()


def load_all():
    data = {}

    data["regime_probs"] = safe_read_csv("regime_probabilities.csv")
    data["garch"] = safe_read_csv("garch.csv")
    data["vix"] = safe_read_csv("vix.csv")
    data["contagion"] = safe_read_csv("contagion.csv")
    data["market_data"] = safe_read_csv("market_data.csv")

    return data

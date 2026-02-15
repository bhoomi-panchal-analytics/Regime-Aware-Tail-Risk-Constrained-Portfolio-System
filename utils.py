import pandas as pd

def load_csv(path):
    return pd.read_csv(path, index_col=0, parse_dates=True)

def load_all():
    data = {}
    data["macro"] = load_csv("data/macro_regime_matrix.csv")
    data["regime_probs"] = load_csv("data/regime_probabilities.csv")
    data["market"] = load_csv("data/market_prices.csv")
    data["vix"] = load_csv("data/vix.csv")
    data["garch"] = load_csv("data/garch_volatility.csv")
    data["contagion"] = load_csv("data/contagion_index.csv")
    data["metrics"] = load_csv("data/portfolio_metrics.csv")
    return data

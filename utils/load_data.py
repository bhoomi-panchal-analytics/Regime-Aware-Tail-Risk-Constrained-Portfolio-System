import pandas as pd
import os

def load_all():
    data = {}

    # Attempt unified market file
    if os.path.exists("data/market_data.csv"):
        data["market_data"] = pd.read_csv(
            "data/market_data.csv",
            index_col=0,
            parse_dates=True
        )
        return data

    # Fallback: try ETF files
    etf_files = [
        "SPY.csv", "TLT.csv", "GLD.csv",
        "DBC.csv", "UUP.csv", "SHY.csv"
    ]

    frames = []

    for file in etf_files:
        path = f"data/{file}"
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df = df.rename(columns={df.columns[0]: file.replace(".csv","")})
            frames.append(df)

    if frames:
        merged = pd.concat(frames, axis=1).dropna()
        data["market_data"] = merged
    else:
        data["market_data"] = pd.DataFrame()

    return data

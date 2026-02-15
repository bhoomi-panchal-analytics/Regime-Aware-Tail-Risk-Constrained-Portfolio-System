import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data")

def safe_load_csv(filename):
    path = os.path.join(DATA_PATH, filename)
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        print(f"WARNING: {filename} not found.")
        return pd.DataFrame()

def load_macro_from_excels():
    macro_df = pd.DataFrame()
    
    for file in os.listdir(DATA_PATH):
        if file.endswith(".xlsx") or file.endswith(".xls"):
            path = os.path.join(DATA_PATH, file)
            df = pd.read_excel(path)
            
            df.columns = [col.strip() for col in df.columns]
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
            df.set_index(df.columns[0], inplace=True)
            
            series_name = file.split(".")[0]
            df.columns = [series_name]
            
            macro_df = pd.concat([macro_df, df], axis=1)
    
    macro_df.sort_index(inplace=True)
    return macro_df

def load_all():
    data = {}
    
    data["macro"] = load_macro_from_excels()
    data["regime_probs"] = safe_load_csv("regime_probabilities.csv")
    data["garch"] = safe_load_csv("garch_volatility.csv")
    data["contagion"] = safe_load_csv("contagion_index.csv")
    data["vix"] = safe_load_csv("vix_synthetic.csv")
    data["metrics"] = safe_load_csv("portfolio_metrics.csv")
    
    return data

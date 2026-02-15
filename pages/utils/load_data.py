import pandas as pd
import os

DATA_PATH = "data"

def load_macro_from_excels():
    macro_df = pd.DataFrame()
    
    for file in os.listdir(DATA_PATH):
        if file.endswith(".xlsx") or file.endswith(".xls"):
            
            path = os.path.join(DATA_PATH, file)
            df = pd.read_excel(path)
            
            # Standardize column names
            df.columns = [col.strip() for col in df.columns]
            
            # Assume first column is date
            df.iloc[:,0] = pd.to_datetime(df.iloc[:,0])
            df.set_index(df.columns[0], inplace=True)
            
            series_name = file.split(".")[0]
            df.columns = [series_name]
            
            macro_df = pd.concat([macro_df, df], axis=1)
    
    macro_df.sort_index(inplace=True)
    return macro_df


def load_all():
    data = {}
    
    data["macro"] = load_macro_from_excels()
    
    # These must already exist as CSV
    data["regime_probs"] = pd.read_csv("data/regime_probabilities.csv", index_col=0, parse_dates=True)
    data["garch"] = pd.read_csv("data/garch_volatility.csv", index_col=0, parse_dates=True)
    data["contagion"] = pd.read_csv("data/contagion_index.csv", index_col=0, parse_dates=True)
    data["vix"] = pd.read_csv("data/vix_synthetic.csv", index_col=0, parse_dates=True)
    data["metrics"] = pd.read_csv("data/portfolio_metrics.csv")
    
    return data

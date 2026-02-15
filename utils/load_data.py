import pandas as pd
import os

def load_all():

    data = {}
    base_path = "data"

    if not os.path.exists(base_path):
        return data

    for file in os.listdir(base_path):

        if file.endswith(".csv"):

            try:
                df = pd.read_csv(os.path.join(base_path, file))

                # Try to detect a date column
                for col in df.columns:
                    try:
                        parsed = pd.to_datetime(df[col])
                        if parsed.notna().sum() > len(df) * 0.8:
                            df[col] = parsed
                            df.set_index(col, inplace=True)
                            break
                    except:
                        continue

                # Ensure datetime index
                df.index = pd.to_datetime(df.index, errors="coerce")

                # Drop invalid dates
                df = df[~df.index.isna()]

                # Sort index
                df = df.sort_index()

                data[file.replace(".csv", "")] = df

            except:
                continue

    return data

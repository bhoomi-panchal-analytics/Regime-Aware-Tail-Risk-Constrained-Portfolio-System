


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
                df = pd.read_csv(
                    os.path.join(base_path, file),
                    index_col=0,
                    parse_dates=True
                )

                data[file.replace(".csv", "")] = df

            except:
                pass

    return data


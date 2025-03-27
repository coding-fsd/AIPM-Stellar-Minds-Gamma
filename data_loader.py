import pandas as pd
import os

def load_all_data(folder_path):
    """Load and combine multiple .dat files."""
    dataframes = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".dat"):
            df = pd.read_csv(os.path.join(folder_path, filename), sep=" ", header=None)
            dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)


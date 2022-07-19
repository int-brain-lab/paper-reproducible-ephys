import pandas as pd
import numpy as np
from reproducible_ephys_functions import save_data_path


def load_dataframeFig5(exists_only=False):
    df_path = save_data_path(figure='figure5').joinpath('figure5_dataframe.csv')
    if exists_only:
        return df_path.exists()
    else:
        if df_path.exists():
            return pd.read_csv(df_path)
        else:
            return None


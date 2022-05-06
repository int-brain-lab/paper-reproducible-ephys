import pandas as pd
import numpy as np
from reproducible_ephys_functions import save_data_path


def load_dataframe(exists_only=False):
    df_path = save_data_path(figure='figure5').joinpath('figure5_dataframe.csv')
    if exists_only:
        return df_path.exists()
    else:
        if df_path.exists():
            return pd.read_csv(df_path)
        else:
            return None


def load_example_neuron(id=614, pid='ce397420-3cd2-4a55-8fd1-5e28321981f4', exists_only=False):
    data_path = save_data_path(figure='figure5').joinpath(f'figure5_example_neuron{id}_{pid}.npy')
    if exists_only:
        return data_path.exists()
    else:
        if data_path.exists():
            return np.load(data_path)
        else:
            return None
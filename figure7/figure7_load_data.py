import pandas as pd
import numpy as np
from reproducible_ephys_functions import save_data_path


def load_dataframe(exists_only=False):
    df_path = save_data_path(figure='figure7').joinpath('figure7_dataframe.csv')
    if exists_only:
        return df_path.exists()
    else:
        if df_path.exists():
            return pd.read_csv(df_path)
        else:
            return None


def load_data(event='move', norm='subtract', smoothing='sliding', exists_only=False):

    smoothing = smoothing or 'none'
    norm = norm or 'none'

    df_path = save_data_path(figure='figure7').joinpath(
        f'figure7_data_event_{event}_smoothing_{smoothing}_norm_{norm}.npz')
    if exists_only:
        return df_path.exists()
    else:
        if df_path.exists():
            return dict(np.load(df_path, allow_pickle=True))
        else:
            return None



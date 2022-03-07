import pandas as pd
import numpy as np
from reproducible_ephys_functions import save_data_path, data_path

def load_dataframe(df_name='chns', exists_only=False):
    df_path = save_data_path(figure='figure3').joinpath(f'figure3_dataframe_{df_name}.csv')
    if exists_only:
        return df_path.exists()
    else:
        if df_path.exists():
            return pd.read_csv(df_path)
        else:
            return None


def load_and_merge_dataframe():
    df_ins = pd.read_csv(save_data_path(figure='figure3').joinpath(f'figure3_dataframe_ins.csv'))
    lfp = pd.read_csv(data_path().joinpath('lfp_ratio_per_region.csv')) # TODO this should eventually be saved in the save path
    data = df_ins.merge(lfp, on=['subject', 'region'])

    return data
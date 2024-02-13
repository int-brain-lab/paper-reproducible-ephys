import pandas as pd
from reproducible_ephys_functions import save_data_path


def load_dataframe(df_name='chns', exists_only=False):
    df_path = save_data_path(figure='fig_ephysfeatures').joinpath(f'fig_ephysfeatures_dataframe_{df_name}.csv')
    if exists_only:
        return df_path.exists()
    else:
        if df_path.exists():
            return pd.read_csv(df_path)
        else:
            return None


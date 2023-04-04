import pandas as pd
import numpy as np
from reproducible_ephys_functions import save_data_path

filtering_criteria = {'min_regions': 0, 'min_lab_region': 3, 'min_rec_lab': 0, 'min_neuron_region': 2, 'freeze': 'release_2022_11'}


#Renamed & remove 'Trial':
tests = {#'trial': 'Trial (first 400 ms)',
          'post_stim': 'Stimulus',
          'post_move': 'Movement period (250 ms)',
          'start_to_move': 'Late reaction period',
          'pre_move': 'Movement initiation',
          'post_reward': 'Reward',
          'pre_move_lr': 'L vs. R pre-movement',
          'avg_ff_post_move': 'Fano Factor'}

def load_dataframe(exists_only=False):
    df_path = save_data_path(figure='fig_taskmodulation').joinpath('fig_taskmodulation_dataframe.csv')
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

    df_path = save_data_path(figure='fig_taskmodulation').joinpath(
        f'fig_taskmodulation_data_event_{event}_smoothing_{smoothing}_norm_{norm}.npz')
    if exists_only:
        return df_path.exists()
    else:
        if df_path.exists():
            return dict(np.load(df_path, allow_pickle=True))
        else:
            return None

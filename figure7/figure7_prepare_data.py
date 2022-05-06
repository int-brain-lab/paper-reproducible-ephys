from figure5.figure5_prepare_data import prepare_data as prepare_data_fig5
from figure7.figure7_load_data import load_dataframe, load_data
from iblutil.numerical import ismember
from one.api import ONE
from reproducible_ephys_functions import get_insertions, save_dataset_info

import numpy as np

default_params = {'fr_bin_size': 0.04,
                  'ff_bin_size': 0.1,
                  'align_event': 'move',
                  'event_epoch': [-0.4, 0.8],
                  'base_event': 'stim',
                  'base_epoch': [-0.4, -0.2],
                  'norm': None,
                  'smoothing': 'sliding',
                  'slide_kwargs_ff': {'n_win': 5, 'causal': 1},
                  'slide_kwargs_fr': {'n_win': 2, 'causal': 1}}


def prepare_data(insertions, one, recompute=False, new_metrics=True, **kwargs):

    align_event = kwargs.get('align_event', default_params['align_event'])
    norm = kwargs.get('norm', default_params['norm'])
    smoothing = kwargs.get('smoothing', default_params['smoothing'])

    if not recompute:
        data_exists = load_data(event=align_event, norm=norm, smoothing=smoothing, exists_only=True)
        if data_exists:
            df = load_dataframe()
            pids = np.array([p['probe_insertion'] for p in insertions])
            isin, _ = ismember(pids, df['pid'].unique())
            if np.all(isin):
                print('Already computed data for set of insertions. Will load in data. To recompute set recompute=True')
                data = load_data()
                return df, data

    df, data = prepare_data_fig5(insertions, one, figure='figure7', recompute=True, new_metrics=new_metrics, **kwargs)
    return df, data


if __name__ == '__main__':
    one = ONE()
    one.record_loaded = True
    insertions = get_insertions(level=2, one=one, freeze='biorxiv_2022_05')
    prepare_data(insertions, one=one, **default_params)
    save_dataset_info(one, figure='figure7')

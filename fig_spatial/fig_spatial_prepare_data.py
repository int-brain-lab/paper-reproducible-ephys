from fig_taskmodulation.fig_taskmodulation_prepare_data import prepare_data as prepare_data_fig_taskmodulation
from fig_spatial.fig_spatial_load_data import load_dataframe, load_data
from iblutil.numerical import ismember
from one.api import ONE
from reproducible_ephys_functions import (get_insertions, save_dataset_info, save_figure_path, BRAIN_REGIONS, combine_regions,
                                          save_data_path, filter_recordings)

from iblatlas.atlas import AllenAtlas, Insertion
from ibllib.pipes.histology import get_brain_regions
import numpy as np
import pandas as pd

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


def prepare_data(insertions, one, recompute=False, **kwargs):

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

    df, data = prepare_data_fig_taskmodulation(insertions, one, figure='fig_spatial', recompute=True, **kwargs)

    df_filt = filter_recordings(df, min_regions=0, min_neuron_region=-1, min_lab_region=0, min_rec_lab=0)
    df_filt = df_filt[df_filt['include'] == 1]

    df_filt.to_csv(save_data_path(figure='fig_spatial').joinpath('fig_spatial_dataframe_filt.csv'))

    save_figure_path(figure='fig_spatial')

    # Compute the centre of mass of the different regions
    planned_ins = one.alyx.rest('trajectories', 'list', probe_insertion=insertions[0]['probe_insertion'], provenance='Planned')[0]
    ba = AllenAtlas()
    ba.compute_surface()
    ins = Insertion.from_dict(planned_ins, brain_atlas=ba)
    xyz = np.c_[ins.tip, ins.entry].T
    brain_regions, _ = get_brain_regions(xyz, brain_atlas=ba)
    rep_site_acro = combine_regions(brain_regions.acronym)

    data = {'region': [], 'x': [], 'y': [], 'z': []}
    for reg in BRAIN_REGIONS:
        reg_idx = np.where(rep_site_acro == reg)[0]
        cent_of_mass_ref = np.mean(brain_regions.xyz[reg_idx], 0)
        data['region'].append(reg)
        data['x'].append(cent_of_mass_ref[0])
        data['y'].append(cent_of_mass_ref[1])
        data['z'].append(cent_of_mass_ref[2])

    df_reg = pd.DataFrame.from_dict(data)
    df_reg.to_csv(save_data_path(figure='fig_spatial').joinpath('fig_spatial_cent_of_mass.csv'))

    return df, data


if __name__ == '__main__':
    one = ONE()
    one.record_loaded = True
    insertions = get_insertions(level=0, recompute=True, one=one, freeze='freeze_2024_03')
    prepare_data(insertions, one=one, recompute=True, **default_params)
    save_dataset_info(one, figure='fig_spatial')

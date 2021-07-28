#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 08:43:23 2021
By: Guido Meijer
"""

from one.api import ONE
from os.path import join
from ibllib.ephys.neuropixel import SITES_COORDINATES
from ibllib.pipes.ephys_alignment import EphysAlignment
from ibllib.atlas import atlas, BrainRegions
from features_2D import psd_data, get_brain_boundaries, plot_probe
from reproducible_ephys_functions import labs, data_path, exclude_recordings
import pandas as pd
import numpy as np


def panel_a(ax, n_rec_per_lab=4, boundary_align='DG-TH', ylim=[-2000, 2000], one=None):
    one = one or ONE()
    brain_atlas = atlas.AllenAtlas(25)
    r = BrainRegions()
    depths = SITES_COORDINATES[:, 1]
    data, lab_colors = probe_plots_data(n_rec_per_lab)

    for iR, (subj, date, probe_label) in enumerate(zip(data['subject'], data['date'], data['probe'])):

        # Download the data and get paths to downloaded data
        eid = one.search(subject=subj, date=date)[0]
        if iR == 0:
            chn_inds = one.load_dataset(eid, dataset=['channels.rawInd.npy'],
                                        collection=f'alf/{probe_label}')
        ephys_path = one.eid2path(eid).joinpath('raw_ephys_data', probe_label)

        insertion = one.alyx.rest('insertions', 'list', session=eid, name=probe_label)
        xyz_picks = np.array(insertion[0].get('json').get('xyz_picks', 0)) / 1e6
        align_key = insertion[0].get('json').get('extended_qc').get('alignment_stored', None)
        trajectory = one.alyx.rest('trajectories', 'list',
                                   provenance='Ephys aligned histology track',
                                   probe_insertion=insertion[0]['id'])
        alignments = trajectory[0]['json']
        feature = np.array(alignments[align_key][0])
        track = np.array(alignments[align_key][1])

        # Instantiate EphysAlignment object
        ephysalign = EphysAlignment(xyz_picks, depths, track_prev=track,
                                    feature_prev=feature,
                                    brain_atlas=brain_atlas)
        xyz_channels = ephysalign.get_channel_locations(feature, track)
        z = xyz_channels[:, 2] * 1e6
        brain_regions = ephysalign.get_brain_locations(xyz_channels)

        boundaries, colours, regions = get_brain_boundaries(brain_regions, z, r)

        if boundary_align is not None:
            z_subtract = boundaries[np.where(np.array(regions) == boundary_align)[0][0] + 1]
            z = z - z_subtract
            boundaries, colours, regions = get_brain_boundaries(brain_regions, z, r)

        # Get LFP data
        plot_data = psd_data(ephys_path, one, eid, chn_inds, freq_range=[20, 80])
        _ = plot_probe(plot_data, z, ax[iR], cmap='viridis')

        ax[iR].set_title(data.loc[iR, 'recording'] + 1,
                         color=lab_colors[data.loc[iR, 'institution']])

        if iR == 0:
            ax[iR].tick_params(axis='y')
            ax[iR].spines["right"].set_visible(False)
            ax[iR].spines["bottom"].set_visible(False)
            ax[iR].spines["top"].set_visible(False)
            ax[iR].set(yticks=np.arange(ylim[0], ylim[1]+1, 500),
                   yticklabels=np.arange(ylim[0], ylim[1]+1, 500) / 1000)
            ax[iR].set_ylabel('Depth relative to DG-Thalamus (mm)')
        else:
            ax[iR].set_axis_off()
        ax[iR].set(ylim=ylim)


def probe_plots_data(n_rec_per_lab=4):
    data = pd.read_csv(join(data_path(), 'metrics_region.csv'))
    data = exclude_recordings(data)
    lab_number_map, institution_map, lab_colors = labs()
    data['institution'] = data.lab.map(institution_map)
    data = data.drop_duplicates(subset='subject')
    data = data.groupby('institution').filter(
        lambda s : s['eid'].unique().shape[0] >= n_rec_per_lab)
    data = data.sort_values(by=['institution', 'subject']).reset_index(drop=True)
    rec_per_lab = data.groupby('institution').size()
    data['recording'] = np.concatenate([np.arange(i) for i in rec_per_lab.values])
    data['lab_position'] = np.linspace(0.18, 0.9, data.shape[0])
    return data, lab_colors




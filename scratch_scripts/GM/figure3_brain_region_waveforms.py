#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 14:22:17 2020

@author: guido
"""

import numpy as np
import pandas as pd
from os.path import join
from pathlib import Path
import brainbox.io.one as bbone
from reproducible_ephys_functions import query, data_path, combine_regions
from oneibl.one import ONE
one = ONE()

# Settings
MIN_SPIKE_AMP = 50
MIN_FR = 0.1
MIN_WAVEFORMS = 20
NEURON_QC = True
DOWNLOAD_DATA = False
REGIONS = ['VISa', 'CA1', 'DG', 'LP', 'PO']
DATA_DIR = data_path()

# Query repeated site trajectories
traj = query()

# %% Loop through repeated site recordings
waveforms_df = pd.DataFrame()
for i in range(len(traj)):
    print('Processing repeated site recording %d of %d' % (i+1, len(traj)))

    # Load in data
    eid = traj[i]['session']['id']
    probe = traj[i]['probe_name']
    lab = traj[i]['session']['lab']

    if DOWNLOAD_DATA:
        _ = one.load(eid, dataset_types=['_phy_spikes_subset.waveforms',
                                         '_phy_spikes_subset.spikes'], download_only=True)
    try:
        spikes, clusters, channels = bbone.load_spike_sorting_with_channel(
            eid, aligned=True, dataset_types=['spikes.amps'], one=one)
        alf_path = one.path_from_eid(eid).joinpath('alf', probe)
        waveforms = np.load(Path(join(alf_path, '_phy_spikes_subset.waveforms.npy')))
        wf_spikes = np.load(Path(join(alf_path, '_phy_spikes_subset.spikes.npy')))

    except Exception as error_message:
        print(error_message)
        continue

    if len(clusters) == 0:
        print('Spike data not found')
        continue

    if 'acronym' not in clusters[probe].keys():
        print('Brain regions not found')
        continue

    # Get neurons that pass QC
    clusters_pass = np.where(clusters[probe]['metrics']['label'] == 1)[0]

    # Loop over regions of interest
    for k, region in enumerate(REGIONS):

        # Get neuron count and firing rate
        region_clusters = [x for x, y in enumerate(combine_regions(clusters[probe]['acronym']))
                           if (region == y) and (x in clusters_pass)]
        neuron_fr = np.empty(len(region_clusters))
        spike_amp = np.empty(len(region_clusters))
        n_waveforms = np.empty(len(region_clusters))
        region_wf = np.empty((82, len(region_clusters)))
        for n, neuron_id in enumerate(region_clusters):

            # Get firing rate
            neuron_fr[n] = (np.sum(spikes[probe]['clusters'] == neuron_id)
                            / np.max(spikes[probe]['times']))

            # Get mean waveform of channel with max amplitude
            n_waveforms[n] = waveforms[spikes[probe].clusters[wf_spikes] == neuron_id].shape[0]
            mean_wf_ch = np.mean(waveforms[spikes[probe].clusters[wf_spikes] == neuron_id], axis=0)
            mean_wf_ch = (mean_wf_ch
                          - np.tile(np.mean(mean_wf_ch, axis=0), (mean_wf_ch.shape[0], 1)))
            mean_wf = mean_wf_ch[:, np.argmin(np.min(mean_wf_ch, axis=0))] * 1000000
            spike_amp[n] = np.abs(np.min(mean_wf) - np.max(mean_wf))
            region_wf[:, n] = mean_wf
        region_wf = region_wf[:, ((neuron_fr > MIN_FR) & (spike_amp > MIN_SPIKE_AMP)
                                  & (n_waveforms >= MIN_WAVEFORMS))]

        # Add to dataframe
        waveforms_df = waveforms_df.append(pd.DataFrame(
               index=[waveforms_df.shape[0] + 1], data={'eid': eid, 'probe': probe, 'lab': lab,
                                                        'region': region,
                                                        'waveforms': [region_wf],
                                                        'n_waveforms': [n_waveforms]}))

# Save result
waveforms_df.to_pickle(join(DATA_DIR, 'figure3_brain_region_waveforms.p'))

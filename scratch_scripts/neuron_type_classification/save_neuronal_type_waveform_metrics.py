#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 15:56:43 2021
By: Guido Meijer
"""

import numpy as np
import pandas as pd
from os.path import join
from scipy.optimize import curve_fit
from brainbox.io.one import SpikeSortingLoader
from brainbox.metrics.single_units import spike_sorting_metrics
from reproducible_ephys_functions import query, save_data_path, combine_regions
from one.api import ONE
from iblatlas.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Settings
NEURON_QC = True
DATA_DIR = save_data_path()
REGIONS = ['PPC', 'CA1', 'DG', 'PO', 'LP']

# Query repeated site trajectories
traj = query()

# %% Definitions


def gaus(x, a, x0, sigma):
    return a * np.exp(-(x-x0)**2 / (2 * sigma**2))


# %% Loop through repeated site recordings
waveforms_df = pd.DataFrame()
for i in range(len(traj)):
    print('Processing repeated site recording %d of %d' % (i+1, len(traj)))

    # Load in data
    eid = traj[i]['session']['id']
    probe = traj[i]['probe_name']
    lab = traj[i]['session']['lab']
    subject = traj[i]['session']['subject']

    # Load in spikes
    try:
        sl = SpikeSortingLoader(eid=eid, pname=probe, one=one, atlas=ba)
        spikes, clusters, channels = sl.load_spike_sorting(revision='2024-03-22')
        clusters = sl.merge_clusters(spikes, clusters, channels)
    except Exception as err:
        print(err)
        continue

    # Load in waveforms
    data = one.load_datasets(eid, datasets=['_phy_spikes_subset.waveforms', '_phy_spikes_subset.spikes',
                                            '_phy_spikes_subset.channels'],
                             collections=[f'alf/{probe}/pykilosort']*3)[0]
    waveforms, wf_spikes, wf_channels = data[0], data[1], data[2]
    waveforms = waveforms * 1000  # to uV

    # Skip recording if data is not present
    if len(clusters) == 0:
        print('Spike data not found')
        continue
    if 'acronym' not in clusters.keys():
        print('Brain regions not found')
        continue
    if 'lateral_um' not in channels.keys():
        print('\nRecording site locations not found, skipping..\n')
        continue

    # Get neurons that pass QC
    if NEURON_QC:
        print('Calculating neuron QC metrics..')
        qc_metrics, _ = spike_sorting_metrics(spikes.times, spikes.clusters, spikes.amps, spikes.depths,
                                              cluster_ids=np.arange(clusters.channels.size))
        clusters_pass = np.where(qc_metrics['label'] == 1)[0]
    else:
        clusters_pass = np.unique(spikes.clusters)
    if len(spikes.clusters) == 0:
        continue

    for n, neuron_id in enumerate(clusters_pass):

        # Get mean waveform of channel with max amplitude
        n_waveforms = waveforms[spikes.clusters[wf_spikes] == neuron_id].shape[0]
        if n_waveforms == 0:
            continue
        mean_wf_ch = np.mean(waveforms[spikes.clusters[wf_spikes] == neuron_id], axis=0)
        mean_wf_ch = (mean_wf_ch
                      - np.tile(np.mean(mean_wf_ch, axis=0), (mean_wf_ch.shape[0], 1)))
        mean_wf = mean_wf_ch[:, np.argmin(np.min(mean_wf_ch, axis=0))]
        wf_time = np.linspace(0, (mean_wf.shape[0] / 30000) * 1000, mean_wf.shape[0])
        spike_amp = np.abs(np.min(mean_wf) - np.max(mean_wf))

        # Get peak-to-trough ratio
        pt_ratio = np.max(mean_wf) / np.abs(np.min(mean_wf))

        # Get part of spike from trough to first peak after the trough
        peak_after_trough = np.argmax(mean_wf[np.argmin(mean_wf):]) + np.argmin(mean_wf)
        repolarization = mean_wf[np.argmin(mean_wf):np.argmax(mean_wf[np.argmin(mean_wf):]) + np.argmin(mean_wf)]

        # Get spike width in ms
        x_time = np.linspace(0, (mean_wf.shape[0] / 30000) * 1000, mean_wf.shape[0])
        peak_to_trough = ((np.argmax(mean_wf) - np.argmin(mean_wf)) / 30000) * 1000
        spike_width = ((peak_after_trough - np.argmin(mean_wf)) / 30000) * 1000

        # Get peak minus through
        pt_subtract = np.max(mean_wf) - np.abs(np.min(mean_wf))

        # Get repolarization slope
        if spike_width <= 0.08:
            continue
        else:
            rp_slope, _, = np.polyfit(x_time[np.argmin(mean_wf):peak_after_trough],
                                      mean_wf[np.argmin(mean_wf):peak_after_trough], 1)

        # Get recovery slope
        rc_slope, _ = np.polyfit(x_time[peak_after_trough:], mean_wf[peak_after_trough:], 1)

        # Get firing rate
        neuron_fr = (np.sum(spikes['clusters'] == neuron_id)
                     / np.max(spikes['times']))

        # Add to dataframe
        waveforms_df = pd.concat((waveforms_df, pd.DataFrame(index=[waveforms_df.shape[0] + 1], data={
            'eid': eid, 'probe': probe, 'lab': lab, 'subject': subject,
            'cluster_id': neuron_id, 'region': clusters.acronym[neuron_id],
            'spike_amp': spike_amp, 'pt_ratio': pt_ratio, 'rp_slope': rp_slope, 'pt_subtract': pt_subtract,
            'rc_slope': rc_slope, 'peak_to_trough': peak_to_trough, 'spike_width': spike_width,
            'firing_rate': neuron_fr, 'n_waveforms': n_waveforms})))

waveforms_df.to_pickle(join(DATA_DIR, 'waveform_metrics.p'))



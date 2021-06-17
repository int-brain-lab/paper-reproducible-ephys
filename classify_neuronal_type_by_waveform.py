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
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
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
FEATURES = ['spike_amp', 'pt_ratio', 'rp_slope', 'spike_width', 'firing_rate']
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
    subject = traj[i]['session']['subject']

    if DOWNLOAD_DATA:
        _ = one.load(eid, dataset_types=['_phy_spikes_subset.waveforms',
                                         '_phy_spikes_subset.spikes',
                                         '_phy_spikes_subset.channels'], download_only=True)
    try:
        spikes, clusters, channels = bbone.load_spike_sorting_with_channel(
            eid, aligned=True, dataset_types=['spikes.amps'], one=one)
        alf_path = one.path_from_eid(eid).joinpath('alf', probe)
        waveforms = np.load(Path(join(alf_path, '_phy_spikes_subset.waveforms.npy')))
        wf_spikes = np.load(Path(join(alf_path, '_phy_spikes_subset.spikes.npy')))
        #wf_channels = np.load(Path(join(alf_path, '_phy_spikes_subset.channels.npy')))

        waveforms = waveforms * 1000000 # to uV

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

    # Loop over clusters
    n_waveforms, spike_amp, pt_ratio, rp_slope, spike_width, neuron_fr = (
        np.empty(len(clusters_pass)), np.empty(len(clusters_pass)), np.empty(len(clusters_pass)),
        np.empty(len(clusters_pass)), np.empty(len(clusters_pass)), np.empty(len(clusters_pass)))
    for n, neuron_id in enumerate(clusters_pass):

        # Get mean waveform of channel with max amplitude
        n_waveforms[n] = waveforms[spikes[probe].clusters[wf_spikes] == neuron_id].shape[0]
        mean_wf_ch = np.mean(waveforms[spikes[probe].clusters[wf_spikes] == neuron_id], axis=0)
        mean_wf_ch = (mean_wf_ch
                      - np.tile(np.mean(mean_wf_ch, axis=0), (mean_wf_ch.shape[0], 1)))
        mean_wf = mean_wf_ch[:, np.argmin(np.min(mean_wf_ch, axis=0))]
        spike_amp[n] = np.abs(np.min(mean_wf) - np.max(mean_wf))

        # Get peak-to-trough ratio
        pt_ratio[n] = np.max(mean_wf) / np.abs(np.min(mean_wf))

        # Get repolarization slope
        if ((np.isnan(mean_wf[0])) or (np.argmin(mean_wf) > np.argmax(mean_wf))
            or (np.abs(np.argmin(mean_wf) - np.argmax(mean_wf)) <= 2)):
            rp_slope[n] = np.nan
        else:
            rp_slope[n] = np.max(np.gradient(mean_wf[
                                    np.argmin(mean_wf):np.argmax(mean_wf)]))

        # Get spike width in ms
        spike_width[n] = (np.abs(np.argmax(mean_wf) - np.argmin(mean_wf)) / 30000) * 1000

        # Get firing rate
        neuron_fr[n] = (np.sum(spikes[probe]['clusters'] == neuron_id)
                        / np.max(spikes[probe]['times']))

        # Get multichannel features
        # To do

    # Add to dataframe
    waveforms_df = waveforms_df.append(pd.DataFrame(data={
        'eid': eid, 'probe': probe, 'lab': lab, 'subject': subject,
        'spike_amp': spike_amp, 'pt_ratio': pt_ratio, 'rp_slope': rp_slope,
        'spike_width': spike_width, 'firing_rate': neuron_fr}))

# Put data in large 2D array
features = waveforms_df[FEATURES].to_numpy()

# Do k-means clustering
km_clusters = KMeans(n_clusters=3, random_state=0).fit(features)

# Do t-SNE embedding for visualization
tsne_embedded = TSNE(n_components=2).fit_transform(features)




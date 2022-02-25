# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 09:32:41 2022

@author: Guido Meijer
"""

import numpy as np
import pandas as pd
from os.path import join
from scipy import signal
from brainbox.io.spikeglx import spikeglx
import brainbox.io.one as bbone
from reproducible_ephys_functions import combine_regions, data_path
from one.api import ONE
one = ONE()

# Settings
OVERWRITE = True
REGIONS = ['PPC', 'CA1', 'DG', 'PO', 'LP']
BASELINE = [-1, 0]
STIM = [0, 1]
LFP_BAND = [20, 80]
WELCH_WIN_LENGTH_SAMPLES = 1024

# Load in PIDs
pids = np.load('repeated_site_pids.npy')

# Load in previously processed data
if OVERWRITE:
    lfp_df = pd.DataFrame()
else:
    lfp_df = pd.read_csv(join(data_path(), 'lfp_ratio_per_region.csv'))

for i, pid in enumerate(pids):

    # Get recording details
    eid, probe = one.pid2eid(pid)
    details = one.get_details(eid)
    subject, lab = details['subject'], details['lab']

    # Skip if already done
    if not OVERWRITE:
        if subject in lfp_df['subject'].values:
            print(f'{subject} already processed')
            continue

    # Load in LFP
    lfp_paths, _ = one.load_datasets(eid, download_only=True, datasets=[
        '_spikeglx_ephysData_g*_t0.imec*.lf.cbin', '_spikeglx_ephysData_g*_t0.imec*.lf.meta',
        '_spikeglx_ephysData_g*_t0.imec*.lf.ch'], collections=[f'raw_ephys_data/{probe}'] * 3)
    lfp_path = lfp_paths[0]
    sr = spikeglx.Reader(lfp_path)

    # Load in trial onset times
    trials = one.load_object(eid, 'trials', attribute=['stimOn_times', 'goCue_times'])

    # Load in channels
    channels = bbone.load_channel_locations(eid, one=one)
    collections = one.list_collections(eid)
    if f'alf/{probe}/pykilosort' in collections:
        collection = f'alf/{probe}/pykilosort'
    else:
        collection = f'alf/{probe}'
    chan_ind = one.load_dataset(eid, dataset='channels.rawInd.npy', collection=collection)

    # Combine regions
    channels[probe]['acronym'] = combine_regions(channels[probe]['acronym'])

    # Loop over trials and get LFP slices for each trial
    this_lfp_df, this_lfp_ch_df = pd.DataFrame(), pd.DataFrame()
    for t, onset in enumerate(trials['stimOn_times'][~np.isnan(trials['stimOn_times'])]):

        # Get LFP slice of current trial
        lfp = sr.read(nsel=slice(int((onset-2) * sr.fs), int((onset+4) * sr.fs), None),
                      csel=slice(None, None, None))[0]
        lfp = lfp.T
        time = np.arange(int((onset-2) * sr.fs), int((onset+4) * sr.fs)) / sr.fs
        Nperseg = int(sr.fs/2)

        # Get baseline power
        f, wb = signal.welch(lfp[np.ix_([True]*lfp.shape[0],
                                        (time > onset + BASELINE[0]) & (time < onset + BASELINE[1]))],
                             fs=sr.fs, window='hanning', nperseg=WELCH_WIN_LENGTH_SAMPLES,
                             detrend='constant', return_onesided=True, scaling='density', axis=-1)

        # Get stim power
        f, ws = signal.welch(lfp[np.ix_([True]*lfp.shape[0],
                                        (time > onset + STIM[0]) & (time < onset + STIM[1]))],
                             fs=sr.fs, window='hanning', nperseg=WELCH_WIN_LENGTH_SAMPLES,
                             detrend='constant', return_onesided=True, scaling='density', axis=-1)
        
        # Get ratio stim / baseline
        wr = ws / wb
        
        # Add all channels to df
        this_lfp_ch_df = this_lfp_ch_df.append(pd.DataFrame(data={
                'lfp_stim_ch': [np.median(ws[:, (f > LFP_BAND[0]) & (f < LFP_BAND[1])], axis=1)[chan_ind]],
                'lfp_bl_ch': [np.median(wb[:, (f > LFP_BAND[0]) & (f < LFP_BAND[1])], axis=1)[chan_ind]],
                'lfp_ratio_ch': [np.median(wr[:, (f > LFP_BAND[0]) & (f < LFP_BAND[1])], axis=1)[chan_ind]],
                'trial': t}))

        # Get LFP power per brain region
        for r, region in enumerate(REGIONS):

            # Get mean ratio for all channels in this region for frequency of interest
            region_chan = chan_ind[channels[probe]['acronym'] == region]
            mean_ratio = np.median(wr[np.ix_(np.isin(np.arange(lfp.shape[0]), region_chan),
                                             (f > LFP_BAND[0]) & (f < LFP_BAND[1]))])
            mean_stim = np.median(ws[np.ix_(np.isin(np.arange(lfp.shape[0]), region_chan),
                                            (f > LFP_BAND[0]) & (f < LFP_BAND[1]))])
            mean_bl = np.median(wb[np.ix_(np.isin(np.arange(lfp.shape[0]), region_chan),
                                          (f > LFP_BAND[0]) & (f < LFP_BAND[1]))])
            
            # Add to dataframe
            this_lfp_df = this_lfp_df.append(pd.DataFrame(index=[lfp_df.shape[0] + 1], data={
                'lfp_ratio': mean_ratio, 'lfp_stim': mean_stim, 'lfp_baseline': mean_bl,
                'region': region, 'subject': subject, 'lab': lab,
                'trial': t}), ignore_index=True)

    # Add to overall dataframe
    median_lfp = this_lfp_df.groupby('region').median()['lfp_ratio']
    median_lfp = median_lfp.reset_index()
    median_lfp['subject'] = subject
    lfp_df = lfp_df.append(median_lfp, ignore_index=True)
    lfp_df.to_csv(join(data_path(), 'lfp_ratio_per_region.csv'), index=False)

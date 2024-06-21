#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 14:22:17 2020

@author: Guido Meijer
"""

import numpy as np
import pandas as pd
from os.path import join
from pathlib import Path
import brainbox.io.one as bbone
from brainbox.metrics.single_units import spike_sorting_metrics
from reproducible_ephys_functions import query, data_path, combine_regions, exclude_recordings
from one.api import ONE
from iblatlas.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# TEMPORARY
one.mode = 'remote'

# Settings
NEURON_QC = True
MIN_NEURONS_PER_REGION = 5
EXCL_REC = True  # Exclude recordings that missed target
DOWNLOAD_WAVEFORMS = False  # Only set to true if you're doing spike waveform analyses
REGIONS = ['PPC', 'CA1', 'DG', 'LP', 'PO']
LFP_BAND_HIGH = [20, 80]
LFP_BAND_LOW = [2, 15]
DATA_DIR = data_path()

# Query repeated site trajectories
if EXCL_REC:
    traj = query()
else:
    traj = query(str_query='probe_insertion__session__projects__name__icontains,ibl_neuropixel_brainwide_01',
                 min_regions=0)

# Initialize dataframe
rep_site = pd.DataFrame()

# %% Loop through repeated site recordings
metrics = pd.DataFrame()
for i in range(len(traj)):
    print('Processing repeated site recording %d of %d' % (i+1, len(traj)))

    # Load in data
    eid = traj[i]['session']['id']
    probe = traj[i]['probe_name']
    pid = traj[i]['probe_insertion']
    lab = traj[i]['session']['lab']
    nickname = traj[i]['session']['subject']
    date = traj[i]['session']['start_time'][:10]

    # Get data collection
    collections = one.list_collections(eid)
    if f'alf/{probe}/pykilosort' in collections:
        alf_path = one.eid2path(eid).joinpath('alf', probe, 'pykilosort')
        collection = f'alf/{probe}/pykilosort'
        print(collection)
    else:
        alf_path = one.eid2path(eid).joinpath('alf', probe)
        collection = f'alf/{probe}'
        print(collection)

    # Download raw ephys data
    try:
        _ = one.load_datasets(eid, datasets=[
            '_iblqc_ephysSpectralDensityAP.freqs.npy', '_iblqc_ephysSpectralDensityAP.power.npy',
            '_iblqc_ephysTimeRmsAP.rms.npy', '_iblqc_ephysSpectralDensityLF.freqs.npy',
            '_iblqc_ephysSpectralDensityLF.power.npy', '_iblqc_ephysTimeRmsLF.rms.npy'],
            collections=[f'raw_ephys_data/{probe}']*6, download_only=True)
    except:
        print('Could not download raw AP and LFP data')
        pass
    try:
        _ = one.load_dataset(eid, dataset='channels.rawInd.npy',
                             collection=collection, download_only=True)
    except:
        print('Could not download channel indices')
        pass
    if DOWNLOAD_WAVEFORMS:
        try:
            _ = one.load_datasets(eid, datasets=['_phy_spikes_subset.waveforms.npy',
                                                 '_phy_spikes_subset.spikes.npy'],
                                  collections=[collection]*2, download_only=True)
        except:
            print('Could not download spike waveforms')
            pass
    try:
        spikes, clusters, channels = bbone.load_spike_sorting_with_channel(
            eid, aligned=True, one=one, brain_atlas=ba)
        ses_path = one.eid2path(eid)
        chn_inds = np.load(Path(join(alf_path, 'channels.rawInd.npy')))
        ephys_path = one.eid2path(eid).joinpath('raw_ephys_data', probe)
        lfp_spectrum = dict()
        lfp_spectrum['freqs'] = np.load(Path(join(ephys_path,
                                                  '_iblqc_ephysSpectralDensityLF.freqs.npy')))
        lfp_spectrum['power'] = np.load(Path(join(ephys_path,
                                                  '_iblqc_ephysSpectralDensityLF.power.npy')))
    except Exception as error_message:
        print(error_message)
        continue

    if (spikes[probe] is None) | (len(clusters) == 0):
        print('Could not load spikes')
        continue

    if (('acronym' not in clusters[probe].keys())):
        print('No histology, skipping session')

    try:
        waveforms = np.load(Path(join(alf_path, '_phy_spikes_subset.waveforms.npy')))
        wf_spikes = np.load(Path(join(alf_path, '_phy_spikes_subset.spikes.npy')))
    except:
        waveforms = []
        wf_spikes = []

    # Get ap band rms
    pid_qc = one.alyx.rest('insertions', 'list', id=pid)[0]['json']['extended_qc']
    if 'apRms_p90_proc' in pid_qc.keys():
        rms_ap_p90 = float(pid_qc['apRms_p90_proc']) * 1e6  # convert to uV
    else:
        rms_ap_p90 = np.nan
    rms_ap = np.load(Path(join(ephys_path, '_iblqc_ephysTimeRmsAP.rms.npy')))
    rms_ap_data = rms_ap * 1e6  # convert to uV
    median = np.mean(np.apply_along_axis(lambda x: np.median(x), 1, rms_ap_data))
    rms_ap_data_median = (np.apply_along_axis(lambda x: x - np.median(x), 1, rms_ap_data)
                          + median)

    # Get neurons that pass QC
    if 'metrics' not in clusters[probe].keys():
        print('No neuron QC found, calculating locally..')
        # Load in spike amplitude and depths
        spikes, clusters, channels = bbone.load_spike_sorting_with_channel(
            eid, aligned=True, one=one, brain_atlas=ba,
            dataset_types=['spikes.amps', 'spikes.depths'])

        # Temporary fix: skip recordings with mismatch in sizes
        if spikes[probe].clusters.shape != spikes[probe].amps.shape:
            print('Shape mismatch between spike attributes, skipping recording!')
            continue

        # Calculate metrics
        qc_metrics, _ = spike_sorting_metrics(spikes[probe].times, spikes[probe].clusters,
                                     spikes[probe].amps, spikes[probe].depths,
                                     cluster_ids=np.arange(clusters[probe].channels.size))
        clusters_pass = np.where(qc_metrics['label'] == 1)[0]
    else:
        clusters_pass = np.where(clusters[probe]['metrics']['label'] == 1)[0]

    # Loop over regions of interest
    for k, region in enumerate(REGIONS):

        # Get neuron count and firing rate
        region_clusters = [x for x, y in enumerate(combine_regions(clusters[probe]['acronym']))
                           if (region == y) and (x in clusters_pass)]

        # Don't calculate single neuron metrics if there are too few neurons
        if len(region_clusters) < MIN_NEURONS_PER_REGION:
            neuron_fr, spike_amp, spike_amp_90, pt_ratio, rp_slope = (
                np.nan, np.nan, np.nan, np.nan, np.nan)
        else:
            neuron_fr = np.empty(len(region_clusters))
            spike_amp = np.empty(len(region_clusters))
            pt_ratio = np.empty(len(region_clusters))
            rp_slope = np.empty(len(region_clusters))
            for n, neuron_id in enumerate(region_clusters):
                # Get firing rate
                neuron_fr[n] = (np.sum(spikes[probe]['clusters'] == neuron_id)
                                / np.max(spikes[probe]['times']))

                try:
                    # Get mean waveform of channel with max amplitude
                    mean_wf_ch = np.mean(waveforms[spikes[probe].clusters[wf_spikes] == neuron_id],
                                         axis=0)
                    mean_wf_ch = (mean_wf_ch
                                  - np.tile(np.mean(mean_wf_ch, axis=0), (mean_wf_ch.shape[0], 1)))
                    mean_wf = mean_wf_ch[:, np.argmin(np.min(mean_wf_ch, axis=0))] * 1000000
                    spike_amp[n] = np.abs(np.min(mean_wf) - np.max(mean_wf))

                    # Get peak-to-trough ration
                    pt_ratio[n] = np.max(mean_wf) / np.abs(np.min(mean_wf))

                    # Get repolarization slope
                    if ((np.isnan(mean_wf[0])) or (np.argmin(mean_wf) > np.argmax(mean_wf))
                        or (np.abs(np.argmin(mean_wf) - np.argmax(mean_wf)) <= 2)):
                        rp_slope[n] = np.nan
                    else:
                        rp_slope[n] = np.max(np.gradient(mean_wf[
                                                np.argmin(mean_wf):np.argmax(mean_wf)]))
                except:
                    spike_amp[n] = np.nan
                    pt_ratio[n] = np.nan
                    rp_slope[n] = np.nan

            # Get mean and 90th percentile of spike amplitudes
            if len(spike_amp) == 0:
                spike_amp_90 = np.nan
            else:
                spike_amp_90 = np.percentile(spike_amp, 95)

        # Get LFP power on high frequencies
        region_chan = chn_inds[[x for x, y in enumerate(combine_regions(channels[probe]['acronym']))
                                if y == region]]
        freqs = ((lfp_spectrum['freqs'] > LFP_BAND_HIGH[0])
                 & (lfp_spectrum['freqs'] < LFP_BAND_HIGH[1]))
        chan_power = lfp_spectrum['power'][:, region_chan]
        lfp_high_region = np.median(10 * np.log(chan_power[freqs]))  # convert to dB

        # Get LFP power on low frequencies
        freqs = ((lfp_spectrum['freqs'] > LFP_BAND_LOW[0])
                 & (lfp_spectrum['freqs'] < LFP_BAND_LOW[1]))
        chan_power = lfp_spectrum['power'][:, region_chan]
        lfp_low_region = np.median(10 * np.log(chan_power[freqs]))  # convert to dB

        # Get AP band rms
        rms_ap_region = np.median(rms_ap_data_median[:, region_chan])

        # Get neuron count
        if len(region_chan) == 0:
            neuron_count = np.nan
        else:
            neuron_count = len(region_clusters)

        # Add to dataframe
        metrics = metrics.append(pd.DataFrame(
                index=[metrics.shape[0] + 1], data={'pid': pid, 'eid': eid, 'probe': probe,
                                                    'lab': lab, 'subject': nickname,
                                                    'region': region, 'date': date,
                                                    'n_channels': region_chan.shape[0],
                                                    'neuron_yield': neuron_count,
                                                    'median_firing_rate': np.median(neuron_fr),
                                                    'mean_firing_rate': np.mean(neuron_fr),
                                                    'spike_amp_mean': np.nanmean(spike_amp),
                                                    'spike_amp_90': spike_amp_90,
                                                    'pt_ratio': np.nanmean(pt_ratio),
                                                    'rp_slope': np.nanmean(rp_slope),
                                                    'lfp_power_low': lfp_low_region,
                                                    'lfp_power_high': lfp_high_region,
                                                    'lfp_band_low': [LFP_BAND_LOW],
                                                    'lfp_band_high': [LFP_BAND_HIGH],
                                                    'rms_ap': rms_ap_region,
                                                    'rms_ap_p90': rms_ap_p90}))

# Save result
print('Saving result..')
if EXCL_REC:
    metrics.to_csv(join(DATA_DIR, 'metrics_region.csv'))

    # Apply additional selection criteria and save list of eids
    metrics_excl, excluded = exclude_recordings(metrics, return_excluded=True)
    np.save('repeated_site_eids.npy', np.array(metrics_excl['eid'].unique(), dtype=str))
    np.save('repeated_site_pids.npy', np.array(metrics_excl['pid'].unique(), dtype=str))
else:
    metrics.to_csv(join(DATA_DIR, 'metrics_region_all.csv'))
print('Done!')

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 19:01:57 2021

@author: Noam Roth
"""



import numpy as np
import pandas as pd
from os.path import join
import alf
from pathlib import Path
import brainbox.io.one as bbone
from reproducible_ephys_functions import query, data_path, combine_regions
from oneibl.one import ONE 
import brainbox as bb
import scipy.stats as stats
one = ONE()

# Settings
MIN_SPIKE_AMP = 50
MIN_FR = 0.1
MAX_AP_RMS = 10000
NEURON_QC = True
DOWNLOAD_DATA = True
#REGIONS = ['VISa', 'CA1', 'DG', 'LP', 'PO']
LFP_BAND_HIGH = [20, 80]
LFP_BAND_LOW = [2, 15]
DATA_DIR = data_path()
pre_time = 0.5 #s to calculate baseline
post_time = 0

# Query repeated site trajectories
traj = query()

# Initialize dataframe
rep_site = pd.DataFrame()

# %% Loop through repeated site recordings
metrics = pd.DataFrame()
for i in range(len(traj)):
    print('Processing repeated site recording %d of %d' % (i+1, len(traj)))

    # Load in data
    eid = traj[i]['session']['id']
    probe = traj[i]['probe_name']
    lab = traj[i]['session']['lab']
    nickname = traj[i]['session']['subject']

    if DOWNLOAD_DATA:
        _ = one.load(eid, dataset_types=['_iblqc_ephysSpectralDensity.freqs',
                                         '_iblqc_ephysSpectralDensity.power',
                                         '_iblqc_ephysTimeRms.rms',
                                         '_phy_spikes_subset.waveforms',
                                         '_phy_spikes_subset.spikes',
                                         'channels.rawInd'], download_only=True)
    try:
        spikes, clusters, channels = bbone.load_spike_sorting_with_channel(
            eid, aligned=True, dataset_types=['spikes.amps'], one=one)
        ses_path = one.path_from_eid(eid)
        alf_path = one.path_from_eid(eid).joinpath('alf', probe)
        chn_inds = np.load(Path(join(alf_path, 'channels.rawInd.npy')))
        ephys_path = one.path_from_eid(eid).joinpath('raw_ephys_data', probe)
        lfp_spectrum = alf.io.load_object(ephys_path, 'ephysSpectralDensityLF',
                                          namespace='iblqc')
        stim_on = one.load(eid, dataset_types=['trials.stimOn_times'])[0]
        stim_on = stim_on[~np.isnan(stim_on)] 
        block_prob = one.load(eid, dataset_types=['trials.probabilityLeft'])
        times_left = stim_on[block_prob[0] == 0.8]
        times_right = stim_on[block_prob[0] == 0.2]
    except Exception as error_message:
        print(error_message)
        continue

    if (('acronym' not in clusters[probe].keys()) | ('metrics' not in clusters[probe].keys())
            | (len(clusters) == 0)):
        print('Missing data, skipping session')
        continue

    try:
        waveforms = np.load(Path(join(alf_path, '_phy_spikes_subset.waveforms.npy')))
        wf_spikes = np.load(Path(join(alf_path, '_phy_spikes_subset.spikes.npy')))
    except:
        
        waveforms = []
        wf_spikes = []

    # Get ap band rms
    rms_ap = alf.io.load_object(ephys_path, 'ephysTimeRmsAP', namespace='iblqc')
    rms_ap_data = rms_ap['rms'] * 1e6  # convert to uV
    median = np.mean(np.apply_along_axis(lambda x: np.median(x), 1, rms_ap_data))
    rms_ap_data_median = (np.apply_along_axis(lambda x: x - np.median(x), 1, rms_ap_data)
                          + median)
    if np.mean(rms_ap_data_median[:, 20]) > MAX_AP_RMS:
        print('AP band RMS too high')
        continue

    # Get neurons that pass QC
    clusters_pass = np.where(clusters[probe]['metrics']['label'] == 1)[0]



    # Get neuron count and firing rate
    clusters_session = [x for x,y in enumerate(combine_regions(clusters[probe]['acronym']))
                       if (x in clusters_pass)]
    neuron_fr = np.empty(len(clusters_session))

    
    #Get baseline fano factor
    activity_preL, _ = bb.singlecell.calculate_peths(spikes[probe].times,
                        spikes[probe].clusters, clusters_session,
                        times_left, pre_time=pre_time, post_time=post_time, 
                        smoothing=0, bin_size=pre_time-post_time)
    activity_preR, _ = bb.singlecell.calculate_peths(spikes[probe].times,
                    spikes[probe].clusters, clusters_session,
                    times_right, pre_time=pre_time, post_time=post_time, 
                    smoothing=0, bin_size=pre_time-post_time)
    
    baselineML = np.mean(activity_preL.means,axis=1)#[activity_preL.means[i][0] for i in range(len(activity_preL.means))]
    baselineVL = np.mean(activity_preL.stds ** 2,axis=1)#[activity_preL.stds[i][0]**2 for i in range(len(activity_preL.means))] 
    baselineMR = np.mean(activity_preL.means,axis=1)#[activity_preR.means[i][0] for i in range(len(activity_preR.means))]
    baselineVR = np.mean(activity_preL.stds ** 2,axis=1)#[activity_preR.stds[i][0]**2 for i in range(len(activity_preR.means))] 


    
    spike_amp = np.empty(len(clusters_session))
    pt_ratio = np.empty(len(clusters_session))
    rp_slope = np.empty(len(clusters_session))
    for n, neuron_id in enumerate(clusters_session):
        # Get firing rate
        neuron_fr[n] = (np.sum(spikes[probe]['clusters'] == neuron_id)
                        / np.max(spikes[probe]['times']))
        



        if len(wf_spikes) > 0:
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
        else:
            spike_amp[n] = np.nan
            pt_ratio[n] = np.nan
            rp_slope[n] = np.nan

    # Impose neuron selection
    # neuron_select = (neuron_fr > MIN_FR) & (spike_amp > MIN_SPIKE_AMP)
    neuron_select = (neuron_fr > MIN_FR)
    neuron_fr = neuron_fr[neuron_select]
    spike_amp = spike_amp[neuron_select]
    neuron_count = np.sum(neuron_select)
    baselineML = np.asarray(baselineML)[neuron_select]
    baselineVL = np.asarray(baselineVL)[neuron_select]
    baselineMR = np.asarray(baselineMR)[neuron_select]
    baselineVR = np.asarray(baselineVR)[neuron_select]
    
    baselineM = baselineML + baselineMR
    baselineV = baselineVL + baselineVL
    
    # Compute fano factor as slope of line fit across neurons
    result = stats.linregress(baselineM ,baselineV)
    FF = result.slope
    
    # Get mean and 90th percentile of spike amplitudes
    if len(spike_amp) == 0:
        spike_amp_90 = np.nan
    else:
        spike_amp_90 = np.percentile(spike_amp, 95)

    # Get LFP power on high frequencies
    chan = chn_inds[[x for x,y in enumerate(combine_regions(channels[probe]['acronym']))]]
                       
    freqs = ((lfp_spectrum['freqs'] > LFP_BAND_HIGH[0])
             & (lfp_spectrum['freqs'] < LFP_BAND_HIGH[1]))
    chan_power = lfp_spectrum['power'][:, chan]
    lfp_high_region = np.mean(10 * np.log(chan_power[freqs]))  # convert to dB

    # Get LFP power on low frequencies
    freqs = ((lfp_spectrum['freqs'] > LFP_BAND_LOW[0])
             & (lfp_spectrum['freqs'] < LFP_BAND_LOW[1]))
    chan_power = lfp_spectrum['power'][:, chan]
    lfp_low_region = np.mean(10 * np.log(chan_power[freqs]))  # convert to dB

    # Get AP band rms
    rms_ap_region = rms_ap_data_median[:, chan].mean()

    # Add to dataframe
    metrics = metrics.append(pd.DataFrame(
            index=[metrics.shape[0] + 1], data={'eid': eid, 'probe': probe, 'lab': lab,
                                                'subject': nickname,
                                                'n_channels': chan.shape[0],
                                                'neuron_yield': neuron_count,
                                                'firing_rate': neuron_fr.mean(),
                                                'baseline_fano_factor': FF,
                                                'spike_amp_mean': np.nanmean(spike_amp),
                                                'spike_amp_90': spike_amp_90,
                                                'pt_ratio': np.nanmean(pt_ratio),
                                                'rp_slope': np.nanmean(rp_slope),
                                                'lfp_power_low': lfp_low_region,
                                                'lfp_power_high': lfp_high_region,
                                                'lfp_band_low': [LFP_BAND_LOW],
                                                'lfp_band_high': [LFP_BAND_HIGH],
                                                'rms_ap': rms_ap_region}))

# Save result
metrics.to_csv(join(DATA_DIR, 'metrics_session.csv'))
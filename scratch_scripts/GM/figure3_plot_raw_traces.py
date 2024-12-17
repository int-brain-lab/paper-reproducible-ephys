#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 15:18:57 2021

@author: guido
"""

import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import brainbox.io.one as bbone
from scipy.signal import filtfilt, butter
from ibllib.pipes.ephys_alignment import EphysAlignment
from scratch_scripts.MF.features_2D import get_brain_boundaries
from ibllib.io import spikeglx
from iblatlas import atlas
from iblatlas.regions import BrainRegions
from ibllib.ephys.neuropixel import SITES_COORDINATES
from reproducible_ephys_paths import FIG_PATH
from oneibl.one import ONE
one = ONE()
depths = SITES_COORDINATES[:, 1]
brain_atlas = atlas.AllenAtlas(25)
r = BrainRegions()

# Settings
PLOT_SEC = 2
RIPPLE_SEC = 0.05
SAMPLING_LF = 2500
SAMPLING_AP = 30000
GLOBAL_REF = False
BAND_PASS_AP = [300, 2000]
BAND_PASS_LF = [0.5, 300]

SUBJECT = 'KS023'
eid = 'aad23144-0e52-4eac-80c5-c4ee2decb198'  # KS023
probe = 'probe01'
PLOT_START_SEC = 610.3
DROP_CH = [9, 11, 19, 22, 24, 73]
RIPPLE_START = 1.57
RIPPLE_CH = np.arange(110, 136)

# Recording
#SUBJECT = 'ZM_2241'
#eid = 'ee40aece-cffd-4edb-a4b6-155f158c666a'  # ZM_2241
# eid = 'f312aaec-3b6f-44b3-86b4-3a0c119c0438'  # CSHL058
#SUBJECT = 'DY_010'
#eid = 'b03fbc44-3d8e-4a6c-8a50-5ea3498568e0'  # DY_010
#probe = 'probe00'

ses_info = one.get_details(eid)

# %% Load in raw data
data_paths = one.load(eid, dataset_types=['ephysData.raw.lf', 'ephysData.raw.ap',
                                          'ephysData.raw.meta', 'channels.rawInd',
                                          'ephysData.raw.ch'],
                      download_only=True)

# %% Load in data

# Load in a slice with a second extra on either side for filtering
raw_lf = spikeglx.Reader(data_paths[int(probe[-1])])
signal_lf = raw_lf.read(nsel=slice(int(SAMPLING_LF * PLOT_START_SEC - SAMPLING_LF),
                                   int(SAMPLING_LF * PLOT_START_SEC
                                       + (SAMPLING_LF * PLOT_SEC + SAMPLING_LF)),
                                   None),
                        csel=slice(None, None, None))[0]
signal_lf = signal_lf.T
raw_ap = spikeglx.Reader(data_paths[int(probe[-1]) + 2])
signal_ap = raw_ap.read(nsel=slice(int(SAMPLING_AP * PLOT_START_SEC - SAMPLING_AP),
                                   int(SAMPLING_AP * PLOT_START_SEC
                                       + (SAMPLING_AP * PLOT_SEC + SAMPLING_AP)),
                                   None),
                        csel=slice(None, None, None))[0]
signal_ap = signal_ap.T

# Load in channel data
_, _, channels = bbone.load_spike_sorting_with_channel(eid, aligned=True, one=one)
alf_path = one.path_from_eid(eid).joinpath('alf', probe)
chn_inds = np.load(Path(join(alf_path, 'channels.rawInd.npy')))

# Apply channel selection
signal_ap = signal_ap[chn_inds, :]
signal_lf = signal_lf[chn_inds, :]

# Load in histology data
insertion = one.alyx.rest('insertions', 'list', session=eid, name=probe)
trajectory = one.alyx.rest('trajectories', 'list', provenance='Ephys aligned histology track',
                           probe_insertion=insertion[0]['id'])
xyz_picks = np.array(insertion[0].get('json').get('xyz_picks', 0)) / 1e6
alignments = trajectory[0]['json']
align_key = insertion[0].get('json').get('extended_qc').get('alignment_stored', None)
feature = np.array(alignments[align_key][0])
track = np.array(alignments[align_key][1])
ephysalign = EphysAlignment(xyz_picks, depths, track_prev=track,
                            feature_prev=feature,
                            brain_atlas=brain_atlas)
xyz_channels = ephysalign.get_channel_locations(feature, track)
brain_regions = ephysalign.get_brain_locations(xyz_channels)
z = xyz_channels[:, 2] * 1e6
boundaries, colours, regions = get_brain_boundaries(brain_regions, z, r)

# Drop channels at the same depth
z, z_index = np.unique(z, return_index=True)
signal_ap = signal_ap[z_index, :]
signal_lf = signal_lf[z_index, :]
acronyms = brain_regions.acronym[z_index]

# Drop specified channels
signal_ap = np.delete(signal_ap, DROP_CH, axis=0)
signal_lf = np.delete(signal_lf, DROP_CH, axis=0)
acronyms = np.delete(acronyms, DROP_CH)

# Apply global reference
if GLOBAL_REF:
    signal_ap = signal_ap - np.mean(signal_ap, axis=0)
    signal_lf = signal_lf - np.mean(signal_lf, axis=0)


def butter_filter(signal, highpass_freq=None, lowpass_freq=None, order=4, fs=2500):

    # The filter type is determined according to the values of cut-off frequencies
    Fn = fs / 2.
    if lowpass_freq and highpass_freq:
        if highpass_freq < lowpass_freq:
            Wn = (highpass_freq / Fn, lowpass_freq / Fn)
            btype = 'bandpass'
        else:
            Wn = (lowpass_freq / Fn, highpass_freq / Fn)
            btype = 'bandstop'
    elif lowpass_freq:
        Wn = lowpass_freq / Fn
        btype = 'lowpass'
    elif highpass_freq:
        Wn = highpass_freq / Fn
        btype = 'highpass'
    else:
        raise ValueError("Either highpass_freq or lowpass_freq must be given")

    # Filter signal
    b, a = butter(order, Wn, btype=btype, output='ba')
    filtered_data = filtfilt(b=b, a=a, x=signal, axis=1)

    return filtered_data


# Zero-mean, amplify and filter traces
for i in range(signal_ap.shape[0]):
    signal_ap[i] = signal_ap[i] - np.mean(signal_ap[i])
    signal_ap[i] = signal_ap[i] * 100000
    signal_ap[i] = butter_filter(signal_ap[i], highpass_freq=300, lowpass_freq=2000,
                                 order=1, fs=SAMPLING_AP)
    signal_lf[i] = signal_lf[i] - np.mean(signal_lf[i])
    signal_lf[i] = signal_lf[i] * 100000
    signal_lf[i] = butter_filter(signal_lf[i], highpass_freq=0.5, lowpass_freq=300, fs=SAMPLING_LF)

# Cut out the second before and after for plotting
signal_ap = signal_ap[:, SAMPLING_AP:-SAMPLING_AP]
signal_lf = signal_lf[:, SAMPLING_LF:-SAMPLING_LF]

# %% Plot AP traces
f, axs = plt.subplots(1, 4, figsize=(15, 10), dpi=300)
these_channels = [i for i, j in enumerate(acronyms) if 'VIS' in j]
these_z = z[these_channels] - np.max(z[these_channels])
for i, chan in enumerate(these_channels):
    axs[0].plot(np.linspace(0, PLOT_SEC * SAMPLING_AP, signal_ap.shape[1]),
                signal_ap[chan] + these_z[i], color='k', lw=0.1)
axs[0].axis('off')
axs[0].set(title='Cortex')

these_channels = [i for i, j in enumerate(acronyms) if 'CA1' in j]
these_z = z[these_channels] - np.max(z[these_channels])
for i, chan in enumerate(these_channels):
    axs[1].plot(np.linspace(0, PLOT_SEC * SAMPLING_AP, signal_ap.shape[1]),
                signal_ap[chan] + these_z[i], color='k', lw=0.1)
axs[1].axis('off')
axs[1].set(title='Hippocampus')

these_channels = [i for i, j in enumerate(acronyms) if 'DG' in j]
these_z = z[these_channels] - np.max(z[these_channels])
for i, chan in enumerate(these_channels):
    axs[2].plot(np.linspace(0, PLOT_SEC * SAMPLING_AP, signal_ap.shape[1]),
                signal_ap[chan] + these_z[i], color='k', lw=0.1)
axs[2].axis('off')
axs[2].set(title='Dentate gyrus')

these_channels = [i for i, j in enumerate(acronyms) if (('LP' in j) or ('PO' in j))]
these_z = z[these_channels] - np.max(z[these_channels])
for i, chan in enumerate(these_channels):
    axs[3].plot(np.linspace(0, PLOT_SEC * SAMPLING_AP, signal_ap.shape[1]),
                signal_ap[chan] + these_z[i], color='k', lw=0.1)
axs[3].axis('off')
axs[3].set(title='Thalamus')

plt.savefig(join(FIG_PATH, f'{SUBJECT}_raw_ap_traces.pdf'))

# %% Plot LF traces
f, axs = plt.subplots(1, 4, figsize=(15, 10), dpi=300)
these_channels = [i for i, j in enumerate(acronyms) if 'VIS' in j]
these_z = z[these_channels] - np.max(z[these_channels])
for i, chan in enumerate(these_channels):
    axs[0].plot(np.linspace(0, PLOT_SEC * SAMPLING_LF, signal_lf.shape[1]),
                signal_lf[chan] + these_z[i], color='k', lw=0.2)
axs[0].axis('off')
axs[0].set(title='Cortex')

these_channels = [i for i, j in enumerate(acronyms) if 'CA1' in j]
these_z = z[these_channels] - np.max(z[these_channels])
for i, chan in enumerate(these_channels):
    axs[1].plot(np.linspace(0, PLOT_SEC * SAMPLING_LF, signal_lf.shape[1]),
                signal_lf[chan] + these_z[i], color='k', lw=0.2)
axs[1].axis('off')
axs[1].set(title='Hippocampus')

these_channels = [i for i, j in enumerate(acronyms) if 'DG' in j]
these_z = z[these_channels] - np.max(z[these_channels])
for i, chan in enumerate(these_channels):
    axs[2].plot(np.linspace(0, PLOT_SEC * SAMPLING_LF, signal_lf.shape[1]),
                signal_lf[chan] + these_z[i], color='k', lw=0.2)
axs[2].axis('off')
axs[2].set(title='Dentate gyrus')

these_channels = [i for i, j in enumerate(acronyms) if (('LP' in j) or ('PO' in j))]
these_z = z[these_channels] - np.max(z[these_channels])
for i, chan in enumerate(these_channels):
    axs[3].plot(np.linspace(0, PLOT_SEC * SAMPLING_LF, signal_lf.shape[1]),
                signal_lf[chan] + these_z[i], color='k', lw=0.2)
axs[3].axis('off')
axs[3].set(title='Thalamus')

plt.savefig(join(FIG_PATH, f'{SUBJECT}_raw_lf_traces.pdf'))

# %% Plot color plot

f, ax1 = plt.subplots(1, 1, figsize=(2, 10), dpi=300)
sns.heatmap(signal_ap, cmap='twilight_shifted', center=0, vmin=-6, vmax=6, cbar=False, ax=ax1)
ax1.axis('off')
plt.savefig(join(FIG_PATH, f'{SUBJECT}_color_probe_plot.png'))

# %% Plot ripple inset

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(3, 5), dpi=300)

these_z = z[RIPPLE_CH] - np.max(z[RIPPLE_CH])
for i, chan in enumerate(RIPPLE_CH):
    ax1.plot(signal_ap[chan, int((RIPPLE_START / PLOT_SEC) * signal_ap.shape[1]):int(((RIPPLE_START + RIPPLE_SEC) / PLOT_SEC) * signal_ap.shape[1])] + these_z[i], color='k', lw=0.2)

    ax2.plot(signal_lf[chan, int((RIPPLE_START / PLOT_SEC) * signal_lf.shape[1]):int(((RIPPLE_START + RIPPLE_SEC) / PLOT_SEC) * signal_lf.shape[1])] + these_z[i], color='k', lw=0.2)

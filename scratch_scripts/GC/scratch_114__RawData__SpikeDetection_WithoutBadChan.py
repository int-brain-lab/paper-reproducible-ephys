'''
View raw data, identify zones with low yield as per spike sorting (pyKS+rapid detection)
'''
# Author: Gaelle, Olivier

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from ibllib.ephys import neuropixel
from ibllib.dsp import voltage
from ibllib.plots import Density
from brainbox.io.one import load_spike_sorting_with_channel, load_spike_sorting_fast
from ibldevtools.Gaelle.Neural_QC_Metric__Yield__SebastianB_2021Oct import yield_metric
from ibldevtools.Gaelle.Neural_QC_Metric__Yield__SebastianB_2021Oct import META_COLORS
from one.api import ONE
from reproducible_ephys_functions import query
from brainbox.io.spikeglx import stream
from iblatlas.regions import BrainRegions
import numpy as np
from ibllib.ephys.spikes import detection
import pandas as pd
from pathlib import Path
import random
import scipy


# from easyqc.gui import viewseis

one = ONE()
br = BrainRegions()
h = neuropixel.trace_header()


# from ibldevtools.Gaelle.scratch_96__RepeatedSite__OverviewPlots -> pid2path
def pid2path(pid, one=None):
    if one is None:
        one = ONE()
    eid, pname = one.pid2eid(pid)
    path = one.eid2path(eid)
    outpath = Path(path).joinpath(pname)
    return outpath


def regions_find_boundaries(channel_regions):
    channel_regions[np.where(channel_regions == None)] = "None"  # Turn into str
    ch1 = channel_regions[1:]
    reg_out = []
    idx_out = []
    for ichan, region in enumerate(ch1):
        if region != channel_regions[ichan]:
            reg_out.append(channel_regions[ichan])
            idx_out.append(ichan)
    reg_out.append(channel_regions[-1])
    return reg_out, idx_out


DETECT_THRESHOLD = -50 * 1e-6  # detecting -50 uV
DISPLAY_TIME = 0.05  # second
V_T0 = [60*10, 60*30, 60*50]  # sample at 10, 30, 50 min in
# DISPLAY_GAIN = -90  # dB

# imageshow boundaries [Volts]
MIN_X = -0.00011
MAX_X = -MIN_X
CMIN = -0.0001
CMAX = 0.00008

XLIM_SPIKCOUNT = 50  # Firing rate
KERNEL_SIZE = 9  # for rolling avg on firing rate

SAMPLE_SKIP = 200  # Skip beginning for show, otherwise blurry due to filter

fig_path = Path.home().joinpath('Desktop/Overview_plot_BWM/Brainwide/Raw_Data_15022022_CortexInvestigate')
# fig_path = Path.home().joinpath('Desktop/Overview_plot_BWM/Repeatedsite/Raw_Data_V2')
fig_path.mkdir(parents=True, exist_ok=True)


# # Get pid used in Repeated site analysis
# PATH_FIG_SAVE = '/Users/gaelle/Desktop/Overview_plot_BWM/Repeatedsite/Raw_Data/'
q = query()
pids_rs = [item['probe_insertion'] for item in q]


# Get pis used in Brainwide map
# PATH_FIG_SAVE = '/Users/gaelle/Desktop/Overview_plot_BWM/Brainwide/Raw_Data/'
django_strg = ['session__projects__name__icontains,ibl_neuropixel_brainwide_01',
               'session__qc__lt,50',
               '~json__qc,CRITICAL',
               'json__extended_qc__tracing_exists,True',  # TODO remove ?
               'session__extended_qc__behavior,1,'
               'session__json__IS_MOCK,False']

django_strg = ['session__projects__name__icontains,ibl_neuropixel_brainwide_01',
               'session__qc__lt,50',
               '~json__qc,CRITICAL',
               'session__extended_qc__behavior,1,'
               'session__json__IS_MOCK,False']

insertions = one.alyx.rest('insertions', 'list', django=django_strg)
pids_all = [item['id'] for item in insertions]

# pids = list(set(pids_all) - set(pids_rs))
pids = pids_rs


# pids = ['f211d54e-e7eb-4188-bdcd-faa9c4317e9e']
random.shuffle(pids)

i_pid = 0
for pid in pids:
    # TEST:  pid = 'c07d13ed-e387-4457-8e33-1d16aed3fd92'

    # get info for later plotting
    path = str(pid2path(pid, one=one))
    path_name = path[33:]  # Note: hardcoded 33 because GC path is `/Users/gaelle/Downloads/FlatIron/`
    # path_name_und = path_name.replace('/', '_')

    i_pid = i_pid + 1
    print(f'------------ {i_pid}/{len(pids)} ------------')

    # 1. GET SPIKE SORTING DATA
    # load data
    eid, pname = one.pid2eid(pid)
    try:
        spikes, clusters, channels = load_spike_sorting_with_channel(eid=eid, one=one, probe=pname,
                                                                     dataset_types=['spikes.amps', 'spikes.depths'])
        '''
        spikes, clusters, channels = load_spike_sorting_fast(eid=eid, one=one, probe=pname,
                                                             # spike_sorter='pykilosort',
                                                             dataset_types=['spikes.amps', 'spikes.depths',
                                                                            'clusters.acronyms'],
                                                             brain_regions=br)
        '''
    except BaseException as e:
        print(f'not able to load spike sorting for pid {pid}')
        continue

    if len(spikes) != 0:
        spikes, clusters, channels = spikes[pname], clusters[pname], channels[pname]
        n_chan = len(channels[list(channels.keys())[0]])

        # compute spike rate per channel overall
        if 'rawInd' not in channels.keys():
            channels['rawInd'] = np.arange(n_chan)
        spikes['rawInd'] = channels['rawInd'][clusters['channels'][spikes['clusters']]]
        nspk_perch_spksort = []
        for i_ch in range(0, n_chan):
            nspk_perch_spksort.append(len(np.where(spikes['rawInd'] == i_ch)[0]))

        # compute yield
        if 'acronym' in clusters.keys():
            yield_SB, channel_regions = yield_metric(clusters, channels, spikes)
            reg_out, idx_out = regions_find_boundaries(channel_regions)
            plot_region = True
        else:
            plot_region = False
    else:
        n_chan = 384  # hardcode
        plot_region = False



    # 2. GET RAW DATA, DETECT SPIKES, AND PLOT FIG
    fig, axs = plt.subplots(nrows=1, ncols=len(V_T0)+1, figsize=(14, 5), gridspec_kw={'width_ratios': [4, 4, 4, 1]})
    fig.suptitle(path_name)
    # init
    spkrate_mat = np.empty((n_chan, len(V_T0)))
    for i_plt, T0 in enumerate(V_T0):        # Catch error if time selected outside of rec boundaries
        ''' GET RAW DATA '''
        try:
            sr, t0 = stream(pid, T0, nsecs=1, one=one)
        except BaseException as e:
            print(f'PID {pid} : recording shorter than {int(T0/60)} min')
            continue
        raw = sr[:, :-1].T
        destripe_plot = voltage.destripe(raw, fs=sr.fs)

        # Detect and remove bad channels prior to spike detection
        labels, xfeats = voltage.detect_bad_channels(raw, sr.fs)
        idx_badchan = np.where(labels != 0)[0]

        # Note: need to pad bad channels with 0 after destriping, not before, otherwide division by 0 -> Nan
        destripe = destripe_plot
        destripe[idx_badchan,:] = 0

        # plot
        X = destripe_plot[:, :int(DISPLAY_TIME * sr.fs)].T
        Xs = X[SAMPLE_SKIP:].T  # Remove artifact at begining
        Tplot = Xs.shape[1]/sr.fs

        # Get total time to compute spike rate for spk sorting output
        T_total = sr.meta['fileSizeBytes'] / sr.nc / 2 / sr.fs
        if len(spikes) != 0:
            spkrate_perch_spksort = np.array(nspk_perch_spksort) / T_total

        ''' DETECTION OF SPIKES ON RAW DATA '''
        detections = detection(destripe.T, fs=sr.fs, h=h, detect_threshold=DETECT_THRESHOLD)
        df_channels = pd.DataFrame(detections).groupby('trace').count()
        # Compute n spikes / spike rate per channel
        nspk_perch = []
        for ich in range(0, n_chan):
            nspk_perch.append(len(np.where(detections['trace'] == ich)[0]))
        spkrate_perch = np.array(nspk_perch) / (destripe.shape[1]/sr.fs)  # TODO rolling median ;
        spkrate_mat[:, i_plt] = spkrate_perch

        ''' PLOT RAW DATA '''
        d = Density(-Xs, fs=sr.fs, taxis=1, ax=axs[i_plt],  vmin=MIN_X, vmax=MAX_X, cmap='Greys')
        axs[i_plt].title.set_text(f'T0 = {int(T0/60)} min')
        axs[i_plt].set_ylabel('')
        axs[i_plt].set_xlim((0, Tplot * 1e3))
        axs[i_plt].set_ylim((0, n_chan))
        # # Plot spikes overlaid
        # axs[i_plt].plot((detections['time'] - SAMPLE_SKIP / sr.fs) * 1e3, detections['trace'],
        #                 'b.', alpha=0.3, markersize=5)

    # Plot brain regions / yield OW / yield SB
    if plot_region:
        idx_pre = [0] + idx_out
        for idx, reg, idx_1 in zip(idx_out+[n_chan], reg_out, idx_pre):
            # Plot region label as text on last subplot
            axs[i_plt+1].plot(0, idx, '_k')
            axs[i_plt+1].set_ylim([0, n_chan])
            idx_str = idx - int((idx-idx_1)/2)

            # Add yield SB onto text
            if reg in yield_SB.keys():
                metric_SB = round(yield_SB[reg][0], 2)
                str_plt_m = reg + f': {metric_SB}'
            else:
                str_plt_m = reg

            # Display text
            axs[i_plt + 1].text(XLIM_SPIKCOUNT, idx_str, str_plt_m)

            # Plot Rectangle overlaid for marking brain regions
            for iax in range(0, len(V_T0)+1):
                if reg != "None":
                    if iax == len(V_T0):
                        width_rect = XLIM_SPIKCOUNT
                    else:
                        width_rect = Tplot*1e3
                    axs[iax].add_patch(Rectangle((0, idx_1), width=width_rect, height=idx-idx_1,
                                                 color=META_COLORS[reg], alpha=0.1,
                                                 edgecolor=None))

    # Plot spike count per channel on last subplot

    # Spike sorting
    if len(spikes) > 0:
        roll_av_spksort = scipy.signal.medfilt(spkrate_perch_spksort, KERNEL_SIZE)
        axs[i_plt + 1].plot(roll_av_spksort, range(0, n_chan), '-r')
    # Detection
    mean_spkrate_perch = spkrate_mat.mean(axis=1)
    roll_av_detection = scipy.signal.medfilt(mean_spkrate_perch, KERNEL_SIZE)
    axs[i_plt + 1].plot(roll_av_detection, range(0, n_chan), '-k', alpha=0.8)
    axs[i_plt + 1].set_ylim((0, n_chan))
    axs[i_plt + 1].set_xlim((0, XLIM_SPIKCOUNT))
    right_side = axs[i_plt + 1].spines["right"]
    right_side.set_visible(False)
    axs[i_plt + 1].set_ylabel('Firing rate (Hz)')

    # Save figure
    fig.tight_layout()
    figname = fig_path.joinpath(f"destripe__{pid}")
    fig.savefig(fname=figname)
    plt.close(fig)


'''
NOTE FOR SELF BELOW
'''
# Plot using sysmic viewer
# eqc_dest = viewseis(X.T, si=1 / sr.fs, h=h, t0=t0, title='destr', taxis=0)
# eqc_dest.ctrl.set_gain(DISPLAY_GAIN)
# eqc_dest.grab().save(PATH_FIG_SAVE + f'raw_dest_{pid}.png')

# sos = scipy.signal.butter(3, 300 / sr.fs / 2, btype='highpass', output='sos')
# butt = scipy.signal.sosfiltfilt(sos, raw)
# show_psd(butt, sr.fs)
# eqc_butt = viewseis(butt[:, :int(DISPLAY_TIME * sr.fs)].T, si=1 / sr.fs, h=h, t0=t0, title='butt', taxis=0)

# Note: To plot all, use:
# # eqc_butt = viewseis(butt.T, si=1 / sr.fs, h=h, t0=t0, title='butt', taxis=0)


''' 
Launch the raw data viewer GUI manually from alignment GUI
Code from instruction here:
Gdoc:  https://docs.google.com/document/d/1fCUcp-QO8x-mgNFJkjAAOAU6cciNFa_LzfOKTvG1IJQ/edit
'''

# from one.api import ONE
# from atlaselectrophysiology.alignment_with_easyqc import viewer
# one = ONE()
# pid = 'e31b4e39-e350-47a9-aca4-72496d99ff2a'
# av = viewer(pid, one=one)

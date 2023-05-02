import one.alf.io as alfio
import matplotlib.pyplot as plt
from pathlib import Path
from ibllib.ephys.ephysqc import EphysQC, phy_model_from_ks2_path, spike_sorting_metrics_ks2
from phylib.io.alf import EphysAlfCreator
from ibllib.pipes.ephys_tasks import SpikeSorting
import numpy as np
from matplotlib import gridspec
from brainbox.processing import bincount2D, compute_cluster_average
from neuropixel import NP2Converter
import spikeglx

LFP_BAND = [20, 80]
MAX_AP_RMS = 40
MAX_LFP_POWER = -150
MIN_NEURONS_PER_CHANNEL = 0.1


def RIGOR_metrics(spikesorting_path, raw_data_path, save_path=None):

    spikesorting_path = Path(spikesorting_path)
    raw_data_path = Path(raw_data_path)

    assert spikesorting_path.exists(), 'Spike sorting path given does not exist'
    assert raw_data_path.exists(), 'Raw data path given does not exist'

    if save_path is None:
        save_path = spikesorting_path.joinpath('RIGOR_ibl')

    save_path.mkdir(exist_ok=True, parents=True)

    ap_file = next(raw_data_path.glob('*ap.*bin'), None)
    assert ap_file, 'Could not find raw ap ephys data file in format .bin or .cbin'

    lf_file = next(raw_data_path.glob('*lf.*bin'), None)
    if lf_file is None:
        print('Raw lfp data not found, computing LFP data from AP data for 600s snippet')
        conv = NP2Converter(ap_file, compress=False)
        conv.init_params(nsamples=int(600 * conv.sr.fs))
        conv._process_NP21(offset=int(conv.sr.ns / 2), assert_shanks=False)

    # Figure out
    ephys_qc = EphysQC(None, session_path=raw_data_path, stream=False, use_alyx=False)
    ephys_qc.probe_path = ephys_qc.session_path
    _ = ephys_qc.run(update=False, ensure=False, out_path=Path(save_path))

    # Compute metric a la IBL
    m = phy_model_from_ks2_path(ks2_path=spikesorting_path, bin_path=raw_data_path)
    ac = EphysAlfCreator(m)
    ac.convert(save_path, label=None, force=True, ampfactor=SpikeSorting._sample2v(ap_file))

    # set depths to spike_depths to catch cases where it can't be computed from pc features (e.g in case of KS3)
    m.depths = np.load(save_path.joinpath('spikes.depths.npy'))
    c = spike_sorting_metrics_ks2(ks2_path=spikesorting_path, m=m)
    c.to_parquet(Path(save_path).joinpath('clusters.metrics.pqt'))

    # If it is four shanks should we split?
    # Read in the metadata, figure out which channels belong to which shank
    meta = spikeglx.read_meta_data(ap_file.with_suffix('.meta'))
    chn_info = spikeglx._map_channels_from_meta(meta)
    n_shanks = np.unique(chn_info['shank']).astype(np.int16)

    if len(n_shanks) > 1:
        for sh in n_shanks:
            chn_idx = np.where(chn_info['shank'] == sh)[0]
            get_metrics(save_path, chn_idx, shank=sh)
    else:
        get_metrics(save_path)


def get_metrics(save_path, channel_idx=None, shank=None):

    # Load in data for shank
    channels = alfio.load_object(save_path, 'channels')
    clusters = alfio.load_object(save_path, 'clusters')
    spikes = alfio.load_object(save_path, 'spikes')

    if shank is not None:
        for k in channels.keys():
            channels[k] = channels[k][channel_idx]

        clusters_shank = np.isin(clusters.channels, channels.rawInd)
        for k in clusters.keys():
            clusters[k] = clusters[k][clusters_shank]

        spikes_shank = np.isin(spikes.clusters, clusters.metrics['cluster_id'])
        for k in spikes.keys():
            spikes[k] = spikes[k][spikes_shank]

    # Compute some metrics
    # AP RMS
    ap = alfio.load_object(save_path, 'ephysChannels')
    ap_rms = np.percentile(ap['apRMS'][1, channels.rawInd], 90) * 1e6

    # LFP PSD
    lfp = alfio.load_object(save_path, 'ephysSpectralDensityLF')
    freqs = (lfp['freqs'] > LFP_BAND[0]) & (lfp['freqs'] < LFP_BAND[1])
    chan_power = lfp['power'][:, channels.rawInd]
    lfp_power = np.mean(10 * np.log(chan_power[freqs]))  # convert to dB

    # NEURON YIELD
    neuron_yield = (clusters.metrics.label == 1).sum() / len(channels.localCoordinates)

    ap_qc = 'PASS' if ap_rms < MAX_AP_RMS else 'FAIL'
    lfp_qc = 'PASS' if lfp_power < MAX_LFP_POWER else 'FAIL'
    yield_qc = 'PASS' if neuron_yield > MIN_NEURONS_PER_CHANNEL else 'FAIL'

    if shank is not None:
        print(f'\n\nMetrics for shank {shank}')
    print(f'AP rms: {ap_rms}, QC: {ap_qc}')
    print(f'LFP power: {lfp_power}, QC: {lfp_qc}')
    print(f'Neuron yield: {neuron_yield}, QC: {yield_qc}')

    # Make a plot with some details
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(1, 1, figure=fig, wspace=0.3, hspace=0.3)
    gs0 = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs[0], width_ratios=[1, 8, 1, 1],
                                           height_ratios=[1, 10], wspace=0.1, hspace=0.3)
    gs0_ax1 = fig.add_subplot(gs0[0, 0])
    gs0_ax2 = fig.add_subplot(gs0[1, 0])
    gs0_ax3 = fig.add_subplot(gs0[0, 1])
    gs0_ax4 = fig.add_subplot(gs0[1, 1])
    gs0_ax5 = fig.add_subplot(gs0[0, 2])
    gs0_ax6 = fig.add_subplot(gs0[1, 2])
    gs0_ax7 = fig.add_subplot(gs0[0, 3])
    gs0_ax8 = fig.add_subplot(gs0[1, 3])

    min_chn = np.min(channels.localCoordinates[:, 1])
    max_chn = np.max(channels.localCoordinates[:, 1])

    # CLUSTER DEPTH VS AMP PLOT
    kp_idx = ~np.isnan(spikes.depths)
    _, cluster_depth, _ = compute_cluster_average(spikes.clusters[kp_idx], spikes.depths[kp_idx])
    _, cluster_amp, _ = compute_cluster_average(spikes.clusters[kp_idx], spikes.amps[kp_idx])
    good_idx = np.where(clusters.metrics.label[np.isin(clusters.metrics.cluster_id,
                                                       np.unique(spikes.clusters[kp_idx]))] == 1)
    mua = gs0_ax2.scatter(cluster_amp * 1e6, cluster_depth, c='r')
    good = gs0_ax2.scatter(cluster_amp[good_idx] * 1e6, cluster_depth[good_idx], c='g')
    gs0_ax1.legend(handles=[mua, good], labels=['mua', 'good'], frameon=False, bbox_to_anchor=(0.8, 0.2))
    gs0_ax1.axis('off')
    gs0_ax2.set_xlabel('Amplitude (uV)')
    gs0_ax2.set_ylabel('Depth along probe')
    gs0_ax2.set_ylim(min_chn, max_chn)

    # SESSION RASTER PLOT
    t_bin = 0.1
    d_bin = 10
    kp_idx = ~np.isnan(spikes.depths)
    session_raster, t_vals, d_vals = bincount2D(spikes.times[kp_idx], spikes.depths[kp_idx],
                                                t_bin, d_bin, ylim=[min_chn, max_chn])
    session_raster = session_raster / t_bin
    gs0_ax4.imshow(session_raster, extent=np.r_[np.min(t_vals), np.max(t_vals), np.min(d_vals), np.max(d_vals)], aspect='auto',
                   origin='lower', vmax=50, cmap='binary')
    gs0_ax3.axis('off')
    gs0_ax4.set_yticks([])
    gs0_ax4.set_xlabel('Time in session')

    # LFP PLOT
    power_vals = 10 * np.log(np.mean(chan_power[freqs], axis=0))[:, np.newaxis]
    clim = np.nanquantile(power_vals, [0.1, 0.9])
    lf_im = gs0_ax6.imshow(power_vals, extent=np.r_[0, 10, min_chn, max_chn], origin='lower', cmap='viridis', aspect='auto',
                           vmin=clim[0], vmax=clim[1])
    cbar = fig.colorbar(lf_im, orientation="horizontal", ax=gs0_ax5)
    cbar.set_label('LFP (dB)')
    ticks = cbar.get_ticks()
    cbar.set_ticks([ticks[0], ticks[-1]])
    gs0_ax5.axis('off')
    gs0_ax6.set_yticks([])
    gs0_ax6.set_xticks([])

    # AP RMS PLOT
    rms_vals = ap['apRMS'][1, :][:, np.newaxis] * 1e6
    clim = np.nanquantile(rms_vals, [0.1, 0.9])
    rms_im = gs0_ax8.imshow(rms_vals, extent=np.r_[0, 10, min_chn, max_chn], origin='lower', cmap='plasma', aspect='auto',
                            vmin=clim[0], vmax=clim[1])
    gs0_ax8.set_yticks([])
    gs0_ax8.set_xticks([])
    cbar = fig.colorbar(rms_im, orientation="horizontal", ax=gs0_ax7)
    cbar.set_label('AP rms (uV)')
    ticks = cbar.get_ticks()
    cbar.set_ticks([ticks[0], ticks[-1]])
    gs0_ax7.axis('off')

    # SAVE PLOT
    plot_path = (save_path.joinpath(f'RIGOR_plot_shank_{shank}.png') if shank is not None
                 else save_path.joinpath('RIGOR_plot.png'))
    print(f'Saving overview plot as {str(plot_path)}')
    fig.savefig(plot_path)


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Offline vs online mode')
    parser.add_argument('-r', '--raw_data_path', required=True, help='Path to raw data folder')
    parser.add_argument('-s', '--spikesorting_path', required=True, help='Path to spike-sorting folder')
    parser.add_argument('-o', '--out_path', default=None, required=False, help='Path to save results')
    args = parser.parse_args()

    RIGOR_metrics(args.spikesorting_path, args.raw_data_path, save_path=args.out_path)

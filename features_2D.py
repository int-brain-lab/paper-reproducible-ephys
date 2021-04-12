# import modules
from oneibl.one import ONE
import matplotlib.pyplot as plt
from matplotlib import cm
import alf.io
import numpy as np
from ibllib.ephys.neuropixel import SITES_COORDINATES
from ibllib.pipes.ephys_alignment import EphysAlignment
from ibllib.atlas import atlas, BrainRegions
from brainbox.processing import bincount2D
from matplotlib.image import NonUniformImage
from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def plot_2D_features(subjects, dates, probes, one=None, brain_atlas=None, freq_range=[2, 15],
                     plot_type='fr', boundary_align=None):

    one = one or ONE()
    brain_atlas = brain_atlas or atlas.AllenAtlas(25)
    r = BrainRegions()
    depths = SITES_COORDINATES[:, 1]

    fig, axs = plt.subplots(1, len(subjects), constrained_layout=False, sharey=True,
                            figsize=(20, 10), dpi=150)
    z_extent = []

    for iR, (subj, date, probe_label) in enumerate(zip(subjects, dates, probes)):
        # Download the data and get paths to downloaded data
        eid = one.search(subject=subj, date=date)[0]
        if iR == 0:
            chn_inds = one.load(eid, dataset_types=['channels.rawInd'])[0]

        ephys_path = one.path_from_eid(eid).joinpath('raw_ephys_data', probe_label)
        alf_path = one.path_from_eid(eid).joinpath('alf', probe_label)

        insertion = one.alyx.rest('insertions', 'list', session=eid, name=probe_label)
        xyz_picks = np.array(insertion[0].get('json').get('xyz_picks', 0)) / 1e6
        align_key = insertion[0].get('json').get('extended_qc').get('alignment_stored', None)
        resolved = insertion[0].get('json').get('extended_qc').get('alignment_resolved', False)
        tracing = insertion[0].get('json').get('extended_qc').get('tracing_exists', False)

        if align_key:
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

            if resolved:
                status = 'resolved'
                col = 'g'
            else:
                status = 'aligned'
                col = 'y'

        elif tracing:

            ephysalign = EphysAlignment(xyz_picks, depths, brain_atlas=brain_atlas)
            feature = ephysalign.feature_init
            track = ephysalign.track_init
            xyz_channels = ephysalign.get_channel_locations(feature, track)
            z = xyz_channels[:, 2] * 1e6
            brain_regions = ephysalign.get_brain_locations(xyz_channels)

            boundaries, colours, regions = get_brain_boundaries(brain_regions, z, r)

            status = 'histology'
            col = 'r'

        else:
            z = depths
            status = 'channels'
            col = 'k'
            boundaries = []

        if boundary_align is not None:
            z_subtract = boundaries[np.where(np.array(regions) == boundary_align)[0][0] + 1]
            z = z - z_subtract
            boundaries, colours, regions = get_brain_boundaries(brain_regions, z, r)

        z_min = np.min(z)
        z_extent.append(z_min)
        z_max = np.max(z)
        z_extent.append(z_max)
        ax = axs[iR]

        if plot_type == 'psd':
            data = psd_data(ephys_path, one, eid, chn_inds, freq_range)
            im = plot_probe(data, z, ax, cmap='viridis')

        elif plot_type == 'rms_ap':
            data = rms_data(ephys_path, one, eid, chn_inds, 'AP')
            im = plot_probe(data, z, ax, cmap='plasma')

        elif plot_type == 'rms_lf':
            data = rms_data(ephys_path, one, eid, chn_inds, 'LF')
            im = plot_probe(data, z, ax, cmap='magma')

        elif plot_type == 'fr_alt':
            data = fr_data_alt(alf_path, one, eid, depths)
            im = plot_probe(data, z, ax, cmap='magma')

        elif plot_type == 'fr':
            y, data = fr_data(alf_path, one, eid, depths)
            y = ephysalign.get_channel_locations(feature, track, y / 1e6)[:, 2] * 1e6
            if boundary_align is not None:
                y = y - z_subtract
            im = NonUniformImage(ax, interpolation='nearest', cmap='magma')
            levels = np.nanquantile(data, [0.1, 0.9])
            im.set_clim(levels[0], levels[1])
            im.set_data(np.array([0]), y, data)
            ax.images.append(im)

        elif plot_type == 'amp':
            y, data = amp_data(alf_path, one, eid, depths)
            y = ephysalign.get_channel_locations(feature, track, y / 1e6)[:, 2] * 1e6
            if boundary_align is not None:
                y = y - z_subtract
            im = NonUniformImage(ax, interpolation='nearest', cmap='magma')
            levels = np.nanquantile(data, [0.1, 0.9])
            im.set_clim(levels[0], levels[1])
            im.set_data(np.array([0]), y, data)
            ax.images.append(im)

        elif plot_type == 'scatter_line':
            x, y, c, s = spike_amp_data(alf_path, eid, one)
            y = ephysalign.get_channel_locations(feature, track, y / 1e6)[:, 2] * 1e6
            if boundary_align is not None:
                y = y - z_subtract
            ax.scatter(x, y, c=c, s=s)

        elif plot_type == 'fr_line':
            y, data = fr_data(alf_path, one, eid, depths)
            y = ephysalign.get_channel_locations(feature, track, y / 1e6)[:, 2] * 1e6
            if boundary_align is not None:
                y = y - z_subtract
            ax.plot(data, y, 'k', linewidth=2)
            ax.set_xlim(0, 50)

        elif plot_type == 'amp_line':
            y, data = amp_data(alf_path, one, eid, depths)
            y = ephysalign.get_channel_locations(feature, track, y / 1e6)[:, 2] * 1e6
            if boundary_align is not None:
                y = y - z_subtract
            ax.plot(data, y, 'k', linewidth=2)
            ax.set_xlim(0, 200)

        elif plot_type == 'regions_line':
            region, region_lab, region_col = region_data(xyz_channels, z, brain_atlas)
            for reg, co, lab in zip(region, region_col, region_lab):
                height = np.abs(reg[1] - reg[0])
                color = co / 255
                ax.bar(x=0.5, height=height, width=1, color=color, bottom=reg[0],
                       edgecolor='w')
                ax.text(x=0.2, y=reg[0] + height/2, s=lab[1], fontdict=None, fontsize=5)

            ax.get_xaxis().set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)

        elif plot_type == 'distance':
            rep_site = {
                'x': -2243,
                'y': -2000,
                'z': 0.0,
                'phi': 180.0,
                'theta': 15.0,
                'depth': 4000
            }
            traj_repeated = atlas.Insertion.from_dict(rep_site).trajectory
            dist = traj_repeated.mindist(xyz_channels) * 1e6
            dist = dist[:, np.newaxis]
            im = NonUniformImage(ax, interpolation='nearest', cmap='Purples')
            levels = [0, 1000]
            im.set_clim(levels[0], levels[1])
            im.set_data(np.array([0]), z, dist)
            ax.images.append(im)

        ax.set_title(subj + '\n' + status, fontsize=8, color=col)
        for bound, col in zip(boundaries, colours):
            ax.hlines(bound, *ax.get_xlim(), linestyles='dashed', linewidth=3, colors=col/255)

    for ax in np.ravel(axs):
        z_max = np.max(z_extent)
        z_min = np.min(z_extent)
        ax.set_ylim(z_min, z_max)

    if plot_type[-4:] != 'line':
        axin = inset_axes(axs[-1], width="25%", height="80%", loc='lower right', borderpad=0,
                         bbox_to_anchor=(0.5, 0.1, 1, 1), bbox_transform=axs[-1].transAxes)
        cbar = fig.colorbar(im, cax=axin, ticks=im.get_clim())
        if plot_type == 'distance':
            cbar.ax.set_yticklabels([f'{levels[0]} um', f'{levels[1]} um'])
        else:
            cbar.ax.set_yticklabels(['10th\nperc.', '90th\nperc'])

    plt.show()
    return fig, axs


def plot_probe(data, z, ax, cmap=None):
    bnk_x, bnk_y, bnk_data = arrange_channels2banks(data, z)
    for x, y, dat in zip(bnk_x, bnk_y, bnk_data):
        im = NonUniformImage(ax, interpolation='nearest', cmap=cmap)
        levels = np.nanquantile(bnk_data[0], [0.1, 0.9])
        im.set_clim(levels[0], levels[1])
        im.set_data(x, y, dat.T)
        ax.images.append(im)
    ax.set_xlim(0, 4.5)
    return im


def psd_data(ephys_path, one, eid, chn_inds, freq_range=[2, 15]):
    try:
        lfp_spectrum = alf.io.load_object(ephys_path, 'ephysSpectralDensityLF',
                                          namespace='iblqc')
    except Exception:
        # Specify the dataset types of interest
        dtypes = ['_iblqc_ephysSpectralDensity.freqs',
                  '_iblqc_ephysSpectralDensity.power']

        _ = one.load(eid, dataset_types=dtypes, download_only=True)
        lfp_spectrum = alf.io.load_object(ephys_path, 'ephysSpectralDensityLF',
                                          namespace='iblqc')

    lfp_freq = lfp_spectrum['freqs']
    lfp_power = lfp_spectrum['power'][:, chn_inds]

    # Define a frequency range of interest
    freq_idx = np.where((lfp_freq >= freq_range[0]) &
                        (lfp_freq < freq_range[1]))[0]

    # Limit data to freq range of interest and also convert to dB
    lfp_spectrum_data = 10 * np.log(lfp_power[freq_idx, :])
    lfp_spectrum_data[np.isinf(lfp_spectrum_data)] = np.nan
    lfp_mean = np.mean(lfp_spectrum_data, axis=0)

    return lfp_mean


def rms_data(ephys_path, one, eid, chn_inds, format):
    try:
        rms_amps = alf.io.load_file_content(Path(ephys_path, '_iblqc_ephysTimeRms' +
                                                 format + '.rms.npy'))
    except Exception:
        dtypes = [
            '_iblqc_ephysTimeRms.rms',
        ]
        _ = one.load(eid, dataset_types=dtypes, download_only=True)
        rms_amps = alf.io.load_file_content(Path(ephys_path, '_iblqc_ephysTimeRms' +
                                                 format + '.rms.npy'))

    rms_avg = (np.mean(rms_amps, axis=0)[chn_inds]) * 1e6

    return rms_avg


def fr_data(alf_path, one, eid, depths):
    try:
        spikes = alf.io.load_object(alf_path, 'spikes')
        spikes['depths']
    except Exception:
        dtypes = [
            'spikes.depths',
            'spikes.amps',
            'spikes.times'
        ]
        _ = one.load(eid, dataset_types=dtypes, download_only=True)
        spikes = alf.io.load_object(alf_path, 'spikes')

    kp_idx = np.where(~np.isnan(spikes['depths']))[0]
    T_BIN = np.max(spikes['times'])
    D_BIN = 20
    nspikes, _, depths = bincount2D(spikes['times'][kp_idx], spikes['depths'][kp_idx],
                                        T_BIN, D_BIN, ylim=[np.min(depths), np.max(depths)])
    mean_fr = nspikes[:, 0] / T_BIN
    mean_fr = mean_fr[:, np.newaxis]

    return depths, mean_fr


def fr_data_alt(alf_path, one, eid, depths):
    try:
        spikes = alf.io.load_object(alf_path, 'spikes')
        clusters = alf.io.load_object(alf_path, 'clusters')
    except Exception:
        dtypes = [
            'spikes.times',
            'spikes.clusters',
            'clusters.channels'
        ]
        _ = one.load(eid, dataset_types=dtypes, download_only=True)
        spikes = alf.io.load_object(alf_path, 'spikes')
        clusters = alf.io.load_object(alf_path, 'clusters')
    spk_chn = clusters.channels[spikes.clusters]
    chn_idx, count = np.unique(spk_chn, return_counts=True)

    fr_chns = np.zeros(depths.shape)
    fr_chns[chn_idx] = count / spikes.times[-1] - spikes.times[0]

    return fr_chns


def amp_data(alf_path, one, eid, depths):
    try:
        spikes = alf.io.load_object(alf_path, 'spikes')
    except Exception:
        dtypes = [
            'spikes.depths',
            'spikes.amps',
            'spikes.times'
        ]
        _ = one.load(eid, dataset_types=dtypes, download_only=True)
        spikes = alf.io.load_object(alf_path, 'spikes')

    kp_idx = np.where(~np.isnan(spikes['depths']))[0]
    T_BIN = np.max(spikes['times'])
    D_BIN = 20

    nspikes, _, _ = bincount2D(spikes['times'][kp_idx], spikes['depths'][kp_idx], T_BIN,
                               D_BIN, ylim=[np.min(depths), np.max(depths)])
    amp, _, depths = bincount2D(spikes['times'][kp_idx], spikes['depths'][kp_idx],
                                T_BIN, D_BIN, ylim=[np.min(depths), np.max(depths)],
                                weights=spikes['amps'][kp_idx])

    mean_amp = np.divide(amp[:, 0], nspikes[:, 0]) * 1e6
    mean_amp[np.isnan(mean_amp)] = 0
    mean_amp = mean_amp[:, np.newaxis]

    return depths, mean_amp

def spike_amp_data(alf_path, one, eid):
    try:
        spikes = alf.io.load_object(alf_path, 'spikes')
    except Exception:
        dtypes = [
            'spikes.depths',
            'spikes.amps',
            'spikes.times'
        ]
        _ = one.load(eid, dataset_types=dtypes, download_only=True)
        spikes = alf.io.load_object(alf_path, 'spikes')

    subsample_factor = 5000
    n_amp_bins = 10
    cmap = 'BuPu'
    amp_range = np.quantile(spikes.amps, [0, 0.9])
    amp_bins = np.linspace(amp_range[0], amp_range[1], n_amp_bins)
    color_bin = np.linspace(0.0, 1.0, n_amp_bins + 1)
    colors = (cm.get_cmap(cmap)(color_bin)[np.newaxis, :, :3][0])

    spike_amps = spikes.amps[0:-1:subsample_factor]
    spike_colors = np.zeros((spike_amps.size, 3))
    spike_size = np.zeros(spike_amps.size)
    for iA in range(amp_bins.size):
        if iA == (amp_bins.size - 1):
            idx = np.where(spike_amps > amp_bins[iA])[0]
            # Make saturated spikes the darkest colour
            spike_colors[idx] = colors[-1]
        else:
            idx = np.where((spike_amps > amp_bins[iA]) & (spike_amps <= amp_bins[iA + 1]))[0]
            spike_colors[idx] = [*colors[iA]]

        spike_size[idx] = iA / (n_amp_bins / 2)

    return spikes.times[0:-1:subsample_factor], spikes.depths[0:-1:subsample_factor], spike_colors, \
           spike_size

def neuron_yield(alf_path):
    return


def region_data(xyz_channels, depths, brain_atlas):
    (region, region_label,
     region_colour, region_id) = EphysAlignment.get_histology_regions(xyz_channels,
                                                                      depths,
                                                                      brain_atlas=brain_atlas)
    return region, region_label, region_colour




def get_brain_boundaries(brain_regions, z, r=None):

    r = r or BrainRegions()
    all_levels = {}
    _brain_id = r.get(ids=brain_regions['id'])
    level = 10
    while level > 0:
        brain_regions = r.get(ids=_brain_id['id'])
        level = np.nanmax(brain_regions.level).astype(int)
        all_levels[f'level_{level}'] = brain_regions['acronym']
        idx = np.where(brain_regions['level'] == level)[0]
        _brain_id = brain_regions
        _brain_id['id'][idx] = _brain_id['parent'][idx]

    boundaries = []
    colours = []
    regions = []
    void = np.where(all_levels['level_3'] == 'void')[0]
    if len(void) > 2:
        boundaries.append(z[void[0]])
        idx = np.where(r.acronym == 'VIS')[0]
        rgb = r.rgb[idx[0]]
        colours.append(rgb)
        regions.append('void-VIS')
    ctx = np.where(all_levels['level_5'] == 'Isocortex')[0]
    if len(ctx) > 2:
        boundaries.append(z[ctx[0]])
        idx = np.where(r.acronym == 'VIS')[0]
        rgb = r.rgb[idx[0]]
        colours.append(rgb)
        regions.append('VIS-HPF')
    hip = np.where(all_levels['level_5'] == 'HPF')[0]
    if len(hip) > 2:
        boundaries.append(z[hip[-1]])
        boundaries.append(z[hip[0]])
        idx = np.where(r.acronym == 'HPF')[0]
        rgb = r.rgb[idx[0]]
        colours.append(rgb)
        colours.append(rgb)
        regions.append('HPF-DG')
    thal = np.where(all_levels['level_2'] == 'BS')[0]
    if len(thal) > 2:
        boundaries.append(z[thal[-1]])
        idx = np.where(r.acronym == 'TH')[0]
        rgb = r.rgb[idx[0]]
        colours.append(rgb)
        regions.append('DG-TH')

    return boundaries, colours, regions


def arrange_channels2banks(data, y):
    bnk_data = []
    bnk_y = []
    bnk_x = []
    for iX, x in enumerate(np.unique(SITES_COORDINATES[:, 0])):
        bnk_idx = np.where(SITES_COORDINATES[:, 0] == x)[0]
        bnk_vals = data[bnk_idx]
        bnk_vals = np.insert(bnk_vals, 0, np.nan)
        bnk_vals = np.append(bnk_vals, np.nan)
        bnk_vals = bnk_vals[:, np.newaxis].T
        bnk_vals = np.insert(bnk_vals, 0, np.full(bnk_vals.shape[1], np.nan), axis=0)
        bnk_vals = np.append(bnk_vals, np.full((1, bnk_vals.shape[1]), np.nan), axis=0)
        bnk_data.append(bnk_vals)

        y_pos = y[bnk_idx]
        y_pos = np.insert(y_pos, 0, y_pos[0] - np.abs(y_pos[2] - y_pos[0]))
        y_pos = np.append(y_pos, y_pos[-1] + np.abs(y_pos[-3] - y_pos[-1]))
        bnk_y.append(y_pos)

        x = np.arange(iX, iX + 3)
        bnk_x.append(x)

    return bnk_x, bnk_y, bnk_data




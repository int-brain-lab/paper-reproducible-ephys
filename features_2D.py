# import modules
from one.api import ONE
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from ibllib.ephys.neuropixel import SITES_COORDINATES
from ibllib.pipes.ephys_alignment import EphysAlignment
from ibllib.atlas import atlas, BrainRegions
from brainbox.processing import bincount2D, compute_cluster_average
from brainbox.metrics.single_units import quick_unit_metrics
from matplotlib.image import NonUniformImage
from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def plot_2D_features(subjects, dates, probes, one=None, brain_atlas=None, freq_range=[20, 80],
                     plot_type='fr', boundary_align=None, show_regions=False, show_boundaries=True,
                     show_colorbar=True, figsize=(20, 10)):

    one = one or ONE()
    brain_atlas = brain_atlas or atlas.AllenAtlas(25)
    r = BrainRegions()
    depths = SITES_COORDINATES[:, 1]

    if len(subjects) == 1:
        fig, axs = plt.subplots(1, len(subjects) + 1, constrained_layout=False, sharey=True,
                                figsize=figsize, dpi=150)
    else:
        fig, axs = plt.subplots(1, len(subjects), constrained_layout=False, sharey=True,
                                figsize=figsize, dpi=150)
    z_extent = []

    for iR, (subj, date, probe_label) in enumerate(zip(subjects, dates, probes)):
        
        # Download the data and get paths to downloaded data
        eid = one.search(subject=subj, task_protocol='ephys', date=date)[0]
        if iR == 0:
            chn_inds = one.load_dataset(eid, dataset=['channels.rawInd.npy'],
                                        collection=f'alf/{probe_label}')

        ephys_path = one.eid2path(eid).joinpath('raw_ephys_data', probe_label)
        collections = one.list_collections(eid)
        if f'alf/{probe_label}/pykilosort' in collections:
            collection = f'alf/{probe_label}/pykilosort'
            print(collection)
        else:
            collection = f'alf/{probe_label}'
            print(collection)

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
            if show_regions:
                bound_reg, col_reg, reg_name = get_brain_boundaries_interest(brain_regions,
                                                                             z, r)

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
            if show_regions:
                bound_reg, col_reg, reg_name = get_brain_boundaries_interest(brain_regions,
                                                                             z, r)

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
            if show_regions:
                bound_reg, col_reg, reg_name = get_brain_boundaries_interest(brain_regions,
                                                                             z, r)

        z_min = np.min(z)
        z_extent.append(z_min)
        z_max = np.max(z)
        z_extent.append(z_max)
        ax = axs[iR]

        if plot_type == 'psd':
            data = psd_data(ephys_path, one, eid, chn_inds, freq_range)
            levels = [-190, -150]
            im = plot_probe(data, z, ax, clim=levels, normalize=False, cmap='viridis')

        elif plot_type == 'rms_ap':
            data = rms_data(ephys_path, one, eid, chn_inds, 'AP')
            im = plot_probe(data, z, ax, cmap='plasma')

        elif plot_type == 'rms_lf':
            data = rms_data(ephys_path, one, eid, chn_inds, 'LF')
            im = plot_probe(data, z, ax, cmap='magma')

        elif plot_type == 'fr_alt':
            data = fr_data_alt(eid, collection, one, depths)
            im = plot_probe(data, z, ax, cmap='magma')

        elif plot_type == 'fr':
            y, data = fr_data(eid, collection, one, depths)
            y = ephysalign.get_channel_locations(feature, track, y / 1e6)[:, 2] * 1e6
            if boundary_align is not None:
                y = y - z_subtract
            im = NonUniformImage(ax, interpolation='nearest', cmap='magma')
            levels = np.nanquantile(data, [0.1, 0.9])
            im.set_clim(levels[0], levels[1])
            im.set_data(np.array([0]), y, data)
            ax.images.append(im)

        elif plot_type == 'amp':
            y, data = amp_data(eid, collection, one, depths)
            y = ephysalign.get_channel_locations(feature, track, y / 1e6)[:, 2] * 1e6
            if boundary_align is not None:
                y = y - z_subtract
            im = NonUniformImage(ax, interpolation='nearest', cmap='magma')
            levels = np.nanquantile(data, [0.1, 0.9])
            im.set_clim(levels[0], levels[1])
            im.set_data(np.array([0]), y, data)
            ax.images.append(im)

        elif plot_type == 'amp_scatter':
            x, y, c = spike_amp_data(eid, collection, one)
            y = ephysalign.get_channel_locations(feature, track, y / 1e6)[:, 2] * 1e6
            if boundary_align is not None:
                y = y - z_subtract
            levels = [0, 30]
            im = ax.scatter(x, y, c=c, s=8, cmap='hot', vmin=levels[0], vmax=levels[1])
            ax.images.append(im)
            ax.set_xlim(1.3, 3)

        elif plot_type == 'fr_line':
            y, data = fr_data(eid, collection, one, depths)
            y = ephysalign.get_channel_locations(feature, track, y / 1e6)[:, 2] * 1e6
            if boundary_align is not None:
                y = y - z_subtract
            ax.plot(data, y, 'k', linewidth=2)
            ax.set_xlim(0, 50)

        elif plot_type == 'amp_line':
            y, data = amp_data(eid, collection, one, depths)
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

        ax.set_title(subj + '\n' + status, color=col)

        if show_regions:
            for reg, co, lab in zip(bound_reg, col_reg, reg_name):
                height = np.abs(reg[1] - reg[0])
                color = co / 255
                width = ax.get_xlim()[1]
                ax.bar(x=width/2, height=height, width=width, color=color, bottom=reg[0],
                       edgecolor='k', alpha=0.5)
                """
                ax.text(x=width/2, y=reg[0] + height / 2, s=lab, fontdict=None, fontsize=10,
                        color='w', fontweight='bold')
                """
        elif show_boundaries:
            for bound, col in zip(boundaries, colours):
                ax.hlines(bound, *ax.get_xlim(), linestyles='dashed',
                          colors=col/255)

    for ax in np.ravel(axs):
        z_max = np.max(z_extent)
        z_min = np.min(z_extent)
        ax.set_ylim(z_min, z_max)

    if show_colorbar & (plot_type[-4:] != 'line'):
        axin = inset_axes(axs[-1], width="25%", height="80%", loc='lower right', borderpad=0,
                         bbox_to_anchor=(0.5, 0.1, 1, 1), bbox_transform=axs[-1].transAxes)
        cbar = fig.colorbar(im, cax=axin, ticks=im.get_clim())
        if plot_type == 'distance':
            cbar.ax.set_yticklabels([f'{levels[0]} um', f'{levels[1]} um'])
        elif plot_type == 'amp_scatter':
            cbar.ax.set_yticklabels([f'{levels[0]} Hz', f'{levels[1]} Hz'])
        elif plot_type == 'psd':
            cbar.ax.set_yticklabels([f'{levels[0]} dB', f'{levels[1]} dB'])
        else:
            cbar.ax.set_yticklabels(['10th\nperc.', '90th\nperc'])
        return fig, axs, cbar
    else:
        return fig, axs



def plot_probe(data, z, ax, clim=[0.1, 0.9], normalize=True, cmap=None):
    bnk_x, bnk_y, bnk_data = arrange_channels2banks(data, z)
    for x, y, dat in zip(bnk_x, bnk_y, bnk_data):
        im = NonUniformImage(ax, interpolation='nearest', cmap=cmap)
        if normalize:
            levels = np.nanquantile(bnk_data[0], clim)
            im.set_clim(levels[0], levels[1])
        else:
            im.set_clim(clim)
        im.set_data(x, y, dat.T)
        ax.images.append(im)
    ax.set_xlim(0, 4.5)
    return im


def psd_data(ephys_path, one, eid, chn_inds, freq_range=[2, 15]):
    try:
        lfp_freq = np.load(ephys_path.joinpath('_iblqc_ephysSpectralDensityLF.freqs.npy'))
        lfp_power = np.load(ephys_path.joinpath('_iblqc_ephysSpectralDensityLF.power.npy'))
    except Exception:
        # Download data
        dtypes = ['_iblqc_ephysSpectralDensityLF.freqs.npy',
                  '_iblqc_ephysSpectralDensityLF.power.npy']

        one.load_datasets(eid, datasets=dtypes,
                          collections=[f'raw_ephys_data/{ephys_path.parts[-1]}'] * 2,
                          download_only=True)
        lfp_freq = np.load(ephys_path.joinpath('_iblqc_ephysSpectralDensityLF.freqs.npy'))
        lfp_power = np.load(ephys_path.joinpath('_iblqc_ephysSpectralDensityLF.power.npy'))

    lfp_power = lfp_power[:, chn_inds]

    # Define a frequency range of interest
    freq_idx = np.where((lfp_freq >= freq_range[0]) &
                        (lfp_freq < freq_range[1]))[0]

    # Limit data to freq range of interest and also convert to dB
    lfp_spectrum_data = 10 * np.log(lfp_power[freq_idx, :])
    lfp_spectrum_data[np.isinf(lfp_spectrum_data)] = np.nan
    lfp_mean = np.mean(lfp_spectrum_data, axis=0)

    return lfp_mean


def rms_data(ephys_path, one, eid, chn_inds, rms_format):
    try:
        rms_amps = np.load(ephys_path.joinpath('_iblqc_ephysTimeRms' + rms_format + '.rms.npy'))
    except Exception:
        one.load_dataset(eid, dataset='_iblqc_ephysTimeRms' + rms_format + '.rms.npy',
                         collection=f'raw_ephys_data/{ephys_path.parts[-1]}', download_only=True)
        rms_amps = np.load(ephys_path.joinpath('_iblqc_ephysTimeRms' + rms_format + '.rms.npy'))

    rms_avg = (np.mean(rms_amps, axis=0)[chn_inds]) * 1e6

    return rms_avg


def fr_data(eid, collection, one, depths):

    spikes = one.load_object(eid, 'spikes', attribute=['depths', 'amps', 'times'],
                             collection=collection)

    kp_idx = np.where(~np.isnan(spikes['depths']))[0]
    T_BIN = np.max(spikes['times'])
    D_BIN = 20
    nspikes, _, depths = bincount2D(spikes['times'][kp_idx], spikes['depths'][kp_idx],
                                        T_BIN, D_BIN, ylim=[np.min(depths), np.max(depths)])
    mean_fr = nspikes[:, 0] / T_BIN
    mean_fr = mean_fr[:, np.newaxis]

    return depths, mean_fr


def fr_data_alt(eid, collection, one, depths):

    spikes = one.load_object(eid, 'spikes', attribute=['times', 'clusters'], collection=collection)
    clusters = one.load_object(eid, 'clusters', attribute=['channels'], collection=collection)

    spk_chn = clusters.channels[spikes.clusters]
    chn_idx, count = np.unique(spk_chn, return_counts=True)

    fr_chns = np.zeros(depths.shape)
    fr_chns[chn_idx] = count / spikes.times[-1] - spikes.times[0]

    return fr_chns


def amp_data(eid, collection, one, depths):

    spikes = one.load_object(eid, 'spikes', attribute=['depths', 'amps', 'times'],
                             collection=collection)

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

def spike_amp_data(eid, collection, one):
    spikes = one.load_object(eid, 'spikes', collection=collection,
                             attribute=['depths', 'amps', 'times', 'clusters'])
    clusters = one.load_object(eid, 'clusters', collection=collection,
                               attribute=['metrics', 'channels'])
    if 'metrics' in clusters.keys():
        clust = np.where(clusters.metrics.label == 1)
    else:
        r = quick_unit_metrics(spikes.clusters, spikes.times, spikes.amps, spikes.depths,
                               cluster_ids=np.arange(clusters.channels.size))
        clust = np.where(r.label == 1)
    spike_idx = np.where(np.isin(spikes['clusters'], clust))[0]

    # compute mean values
    cluster, cluster_depth, n_cluster = compute_cluster_average(spikes.clusters[spike_idx],
                                                                spikes.depths[spike_idx])
    _, cluster_amp, _ = compute_cluster_average(spikes.clusters[spike_idx],
                                                spikes.amps[spike_idx])
    cluster_amp = cluster_amp * 1e6
    cluster_amp = np.log10(cluster_amp)
    cluster_fr = n_cluster / (np.max(spikes.times[spike_idx]) - np.min(spikes.times[spike_idx]))

    return cluster_amp, cluster_depth, cluster_fr

def neuron_yield(alf_path):
    return


def region_data(xyz_channels, depths, brain_atlas):
    (region, region_label,
     region_colour, region_id) = EphysAlignment.get_histology_regions(xyz_channels,
                                                                      depths,
                                                                      brain_atlas=brain_atlas)
    return region, region_label, region_colour



def get_brain_boundaries_interest(brain_regions, z, r=None):
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

    br_acro = ['VISa', 'VISam', 'DG', 'CA1', 'LP', 'PO']
    br_level = [7, 7, 7, 8, 7, 7]
    for acro, lev in zip(br_acro, br_level):
        acr = np.where(all_levels[f'level_{lev}'] == acro)[0]
        if len(acr) > 2:
            boundaries.append([z[acr[0]], z[acr[-1]]])
            idx = np.where(r.acronym == acro)[0]
            rgb = r.rgb[idx[0]]
            colours.append(rgb)
            regions.append(acro)

    return boundaries, colours, regions

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




# import modules
from oneibl.one import ONE
import matplotlib.pyplot as plt
import alf.io
import numpy as np
from ibllib.ephys.neuropixel import SITES_COORDINATES
from ibllib.pipes.ephys_alignment import EphysAlignment
from ibllib.atlas import atlas, regions_from_allen_csv
from brainbox.processing import bincount2D
from matplotlib.image import NonUniformImage
import matplotlib.pyplot as plt

# Specify the dataset types of interest
dtypes = ['_iblqc_ephysSpectralDensity.freqs',
          '_iblqc_ephysSpectralDensity.power']

chn_inds = np.load('C:/Users/Mayo/Downloads/FlatIron/churchlandlab/Subjects/CSHL049/2020-01-08/'
                   '001/alf/probe00/channels.rawInd.npy')


def plot_2D_features(subjects, dates, probes, one=None, brain_atlas=None):
    plot_type = 'psd'
    one = one or ONE()
    brain_atlas = brain_atlas or atlas.AllenAtlas(25)
    r = regions_from_allen_csv()
    depths = SITES_COORDINATES[:, 1]

    fig, axs = plt.subplots(1, len(subjects), constrained_layout=True, sharey=True)
    z_extent = []

    for iR, (subj, date, probe_label) in enumerate(zip(subjects, dates, probes)):
        print(iR)
        # Download the data and get paths to downloaded data
        eid = one.search(subject=subj, date=date)[0]
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

            boundaries, colours = get_brain_boundaries(brain_regions, z, r)

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

            boundaries, colours = get_brain_boundaries(brain_regions, z, r)

            status = 'histology'
            col = 'r'

        else:
            z = depths
            status = 'channels'
            col = 'k'
            boundaries = []

        z_min = np.min(z)
        z_extent.append(z_min)
        z_max = np.max(z)
        z_extent.append(z_max)
        ax = axs[iR]

        if plot_type == 'psd':
            x, vals = psd_data(ephys_path, one, eid)
            y = z
            vals = np.insert(vals, 0, np.full(vals.shape[0], np.nan), axis=1)
            vals = np.append(vals, np.full((vals.shape[0], 1), np.nan), axis=1)
            y = np.insert(y, 0, y[0] - np.abs(y[2] - y[0]))
            y = np.append(y, y[-1] + np.abs(y[-3] - y[-1]))
            im = NonUniformImage(ax, interpolation='nearest', extent=(0, 300, z_min, z_max),
                                 cmap='viridis')
            levels = np.nanquantile(vals, [0.1, 0.9])
            im.set_clim(levels[0], levels[1])
            im.set_data(x, y, vals.T)
            ax.images.append(im)
            ax.set_xlim(0, 300)

        elif plot_type == 'fr':
            _, y, vals = fr_data(alf_path, one, eid, depths)
            y = ephysalign.get_channel_locations(feature, track, y / 1e6)[:, 2] * 1e6
            ax.plot(vals, y, 'k', linewidth=2)
            ax.set_xlim(0, 50)

        ax.set_title(subj + '\n' + status, fontsize=8, color=col)
        for bound, col in zip(boundaries, colours):
            ax.hlines(bound, *ax.get_xlim(), linestyles='dashed', linewidth=3, colors=col/255)

    for ax in axs:
        z_max = np.max(z_extent)
        z_min = np.min(z_extent)
        ax.set_ylim(z_min, z_max)

    plt.show()


def psd_data(ephys_path, one, eid):
    try:
        lfp_spectrum = alf.io.load_object(ephys_path, 'ephysSpectralDensityLF',
                                          namespace='iblqc')
    except Exception:
        _ = one.load(eid, dataset_types=dtypes, download_only=True)
        lfp_spectrum = alf.io.load_object(ephys_path, 'ephysSpectralDensityLF',
                                          namespace='iblqc')

    lfp_freq = lfp_spectrum['freqs']
    lfp_power = lfp_spectrum['power'][:, chn_inds]

    # Define a frequency range of interest
    freq_range = [0, 300]
    freq_idx = np.where((lfp_freq >= freq_range[0]) &
                        (lfp_freq < freq_range[1]))[0]
    freqs = lfp_freq[freq_idx]

    # Limit data to freq range of interest and also convert to dB
    lfp_spectrum_data = 10 * np.log(lfp_power[freq_idx, :])
    lfp_spectrum_data[np.isinf(lfp_spectrum_data)] = np.nan

    return freqs, lfp_spectrum_data


def fr_data(alf_path, one, eid, depths):
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
    nspikes, times, depths = bincount2D(spikes['times'][kp_idx], spikes['depths'][kp_idx],
                                        T_BIN, D_BIN, ylim=[np.min(depths), np.max(depths)])
    mean_fr = nspikes[:, 0] / T_BIN

    return times, depths, mean_fr


def get_brain_boundaries(brain_regions, z, r=None):

    r = r or regions_from_allen_csv()
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
    void = np.where(all_levels['level_3'] == 'void')[0]
    if len(void) > 2:
        boundaries.append(z[void[0]])
        idx = np.where(r.acronym == 'VIS')[0]
        rgb = r.rgb[idx[0]]
        colours.append(rgb)

    ctx = np.where(all_levels['level_5'] == 'Isocortex')[0]
    if len(ctx) > 2:
        boundaries.append(z[ctx[0]])
        idx = np.where(r.acronym == 'VIS')[0]
        rgb = r.rgb[idx[0]]
        colours.append(rgb)
    hip = np.where(all_levels['level_5'] == 'HPF')[0]
    if len(hip) > 2:
        boundaries.append(z[hip[-1]])
        boundaries.append(z[hip[0]])
        idx = np.where(r.acronym == 'HPF')[0]
        rgb = r.rgb[idx[0]]
        colours.append(rgb)
        colours.append(rgb)
    thal = np.where(all_levels['level_2'] == 'BS')[0]
    if len(thal) > 2:
        boundaries.append(z[thal[-1]])
        idx = np.where(r.acronym == 'TH')[0]
        rgb = r.rgb[idx[0]]
        colours.append(rgb)

    return boundaries, colours
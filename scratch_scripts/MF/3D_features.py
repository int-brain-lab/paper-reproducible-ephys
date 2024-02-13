from oneibl.one import ONE
from ibllib.ephys.neuropixel import TIP_SIZE_UM, SITES_COORDINATES
from ibllib.pipes import histology
from ibllib.pipes.ephys_alignment import EphysAlignment
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
import iblatlas.atlas as atlas
import numpy as np
from mayavi import mlab
import alf.io
from brainbox.processing import bincount2D
from atlaselectrophysiology import rendering
from pathlib import Path
from reproducible_ephys_functions import query
one = ONE()
ba = atlas.AllenAtlas(25)

trajectories = query()

probe_insertion = [t['probe_insertion'] for t in trajectories]
depths = SITES_COORDINATES[:, 1] / 1e6
_, depth_idx = np.unique(SITES_COORDINATES[:, 1], return_index=True)

data_type = 'LFP Spectrum'
# Gather all the data together
is_aligned = []
for ip, p in enumerate(probe_insertion):
    insertion = one.alyx.rest('insertions', 'list', id=p)[0]
    if insertion.get('json'):
        xyz_picks = np.array(insertion['json']['xyz_picks']) / 1e6
    else:
        print('no xyz')
        continue

    traj = one.alyx.rest('trajectories', 'list', provenance='Ephys aligned histology track',
                         probe_insertion=p)
    if len(traj) != 0:
        prev_align = [*traj[0]['json'].keys()]
        # To make sure they are ordered by date added, default to latest fit
        prev_align = sorted(prev_align, reverse=True)
        feature = np.array(traj[0]['json'][prev_align[0]][0])
        track = np.array(traj[0]['json'][prev_align[0]][1])
        ephys_align = EphysAlignment(xyz_picks, chn_depths=depths, track_prev=track,
                                     feature_prev=feature, brain_atlas=ba)
        aligned = True
    else:
        ephys_align = EphysAlignment(xyz_picks, chn_depths=depths, brain_atlas=ba)
        feature, track, _ = ephys_align.get_track_and_feature()
        aligned = False

    channel_depths_track = ephys_align.feature2track(depths, feature, track) - \
                           ephys_align.track_extent[0]
    # For smooth track
    xyz_channels = histology.interpolate_along_track(ephys_align.xyz_track[[0, -1], :],
                                                     channel_depths_track)
    mlapdv = ba.xyz2ccf(xyz_channels[depth_idx, :])

    eid = insertion['session']
    probe = insertion['name']
    alf_path = one.path_from_eid(eid).joinpath('alf', probe)
    ephys_path = one.path_from_eid(eid).joinpath('raw_ephys_data', probe)
    if ip == 0:
        chn_ind = one.load(eid, dataset_types='channels.rawInd')[0]

    try:
        if data_type == 'Firing Rate':
            spikes = load_spike_data(eid, alf_path)
            avg_data = avg_fr_along_probe(spikes, depths * 1e6)

        if data_type == 'Amplitude':
            spikes = load_spike_data(eid, alf_path)
            avg_data = avg_amp_along_probe(spikes, depths * 1e6)

        if data_type == 'RMS LFP':
            rms_lf = load_rms_data(eid, ephys_path, 'LF')
            avg_data = rms_along_probe(rms_lf, depths, chn_ind)

        if data_type == 'RMS AP':
            rms_ap = load_rms_data(eid, ephys_path, 'AP')
            avg_data = rms_along_probe(rms_ap, depths, chn_ind)

        if data_type == 'LFP Spectrum':
            lfp_power, lfp_freq = load_lfp_data(eid, ephys_path)
            avg_data = lfp_along_probe(lfp_power, lfp_freq, depths, chn_ind)

        if ip == 0:
            mlapdv_all = mlapdv
            data_all = avg_data
            is_aligned.append(aligned)
        else:
            mlapdv_all = np.dstack([mlapdv_all, mlapdv])
            data_all = np.vstack([data_all, avg_data])
            is_aligned.append(aligned)
    except Exception as err:
        print(err)
        continue


# Now plot everything
#min_val, max_val = np.quantile(data_all, [0.1, 0.9])

fig = mlab.figure(bgcolor=(1, 1, 1))
data_all[np.isinf(data_all)] = np.nan
for iT in np.where(is_aligned)[0].tolist():
    min_val, max_val = np.nanquantile(data_all[iT, :], [0.1, 0.9])
    mlab.plot3d(mlapdv_all[:, 1, iT], mlapdv_all[:, 2, iT], mlapdv_all[:, 0, iT],
                data_all[iT, :], line_width=1, tube_radius=40, colormap='viridis',
                vmin=min_val, vmax=max_val)

# for iT in range(data_all.shape[0]):
#     min_val, max_val = np.quantile(data_all[iT, :], [0.1, 0.9])
#     mlab.plot3d(mlapdv_all[:, 1, iT], mlapdv_all[:, 2, iT], mlapdv_all[:, 0, iT],
#                 data_all[iT, :], line_width=1, tube_radius=40, colormap='plasma',
#                 vmin=min_val, vmax=max_val)

# Get the planned regions repeated site should pass through and download the structure meshes
ins_planned = atlas.Insertion.from_dict(trajectories[0])
xyz_planned = histology.interpolate_along_track(ins_planned.xyz, depths + TIP_SIZE_UM / 1e6)
brain_regions = ba.regions.get(ba.get_labels(xyz_planned))
acronym, idx = np.unique(brain_regions['acronym'], return_index=True)
idx = idx[:11]

mcc = MouseConnectivityCache(resolution=25)
for id in idx:
    # _ = mcc.get_structure_mesh(b_id)
    b_id = brain_regions['id'][id]
    file_path = mcc.get_cache_path(None, mcc.STRUCTURE_MESH_KEY, mcc.reference_space_key, b_id)
    color = tuple(brain_regions['rgb'][id] / 255)
    rendering.add_mesh(fig, file_path, color, opacity=0.2)

root_id = 205
b_id = brain_regions['id'][root_id]
file_path = mcc.get_cache_path(None, mcc.STRUCTURE_MESH_KEY, mcc.reference_space_key, b_id)
color = tuple(brain_regions['rgb'][root_id] / 255)
rendering.add_mesh(fig, file_path, color, opacity=0.2)

mlab.view(azimuth=180, elevation=0)
mlab.view(azimuth=210, elevation=210, reset_roll=False)


cb = mlab.colorbar(title=data_type, orientation='vertical', nb_labels=3, label_fmt='%.1f')
cb.data_range = (min_val, max_val)
cb.label_text_property.color = (0., 0., 0.)
cb.scalar_bar.unconstrained_font_size = True
cb.label_text_property.font_size = 12
cb.title_text_property.color = (0., 0., 0.)
cb.scalar_bar.position = np.array([0.01,  0.15])
cb.scalar_bar.position2 = np.array([0.05,  0.7])


def load_spike_data(eid, alf_path):
    dtypes = [
        'spikes.depths',
        'spikes.amps',
        'spikes.times'
    ]

    _ = one.load(eid, dataset_types=dtypes, download_only=True)
    spikes = alf.io.load_object(alf_path, 'spikes')
    return spikes


def load_lfp_data(eid, ephys_path):
    dtypes = [
        '_iblqc_ephysSpectralDensity.freqs',
        '_iblqc_ephysSpectralDensity.power'
    ]
    _ = one.load(eid, dataset_types=dtypes, download_only=True)
    lfp_spectrum = alf.io.load_object(ephys_path, 'ephysSpectralDensityLF', namespace='iblqc')
    lfp_freq = lfp_spectrum.get('freqs')
    lfp_power = lfp_spectrum.get('power')

    return lfp_power, lfp_freq


def load_rms_data(eid, ephys_path, format):
    dtypes = [
        '_iblqc_ephysTimeRms.rms',
    ]
    _ = one.load(eid, dataset_types=dtypes, download_only=True)
    rms_amps = alf.io.load_file_content(Path(ephys_path, '_iblqc_ephysTimeRms' +
                                             format + '.rms.npy'))
    return rms_amps


def avg_fr_along_probe(spikes, chn_depths, D_BIN=20):
    kp_idx = np.where(~np.isnan(spikes['depths']))[0]
    T_BIN = np.max(spikes['times'])
    nspikes, times, depths = bincount2D(spikes['times'][kp_idx], spikes['depths'][kp_idx], T_BIN,
                                        D_BIN, ylim=[np.min(chn_depths), np.max(chn_depths)])
    mean_fr = nspikes[:, 0] / T_BIN

    return mean_fr


def avg_amp_along_probe(spikes, chn_depths, D_BIN=20):
    kp_idx = np.where(~np.isnan(spikes['depths']))[0]
    T_BIN = np.max(spikes['times'])
    nspikes, _, _ = bincount2D(spikes['times'][kp_idx], spikes['depths'][kp_idx], T_BIN,
                                D_BIN, ylim=[np.min(chn_depths), np.max(chn_depths)])
    amp, times, depths = bincount2D(spikes['times'][kp_idx], spikes['depths'][kp_idx],
                                    T_BIN, D_BIN, ylim=[np.min(chn_depths), np.max(chn_depths)],
                                    weights=spikes['amps'][kp_idx])

    mean_amp = np.divide(amp[:, 0], nspikes[:, 0]) * 1e6
    mean_amp[np.isnan(mean_amp)] = 0
    remove_bins = np.where(nspikes[:, 0] < 50)[0]
    mean_amp[remove_bins] = 0
    return mean_amp


def rms_along_probe(rms, chn_depths, chn_ind):

    _rms = np.take(rms_ap, chn_ind, axis=1) * 1e6

    rms_avg = np.mean(avg_across_depth(_rms, chn_depths), axis=0)

    return rms_avg


def lfp_along_probe(lfp_power, lfp_freq, chn_depths, chn_ind, freq_range=[0, 100]):

    freq_idx = np.where((lfp_freq >= freq_range[0]) &
                        (lfp_freq < freq_range[1]))[0]
    _lfp = np.take(lfp_power[freq_idx], chn_ind, axis=1)
    _lfp_dB = 10 * np.log10(_lfp)

    avg_lfp = np.mean(avg_across_depth(_lfp_dB, chn_depths), axis=0)

    return avg_lfp


def avg_across_depth(data, chn_depths):

    _, chn_idx, chn_count = np.unique(chn_depths, return_index=True, return_counts=True)
    chn_idx_eq = np.copy(chn_idx)
    chn_idx_eq[np.where(chn_count == 2)] += 1

    def avg_chn_depth(a):
        return np.mean([a[chn_idx], a[chn_idx_eq]], axis=0)

    return np.apply_along_axis(avg_chn_depth, 1, data)




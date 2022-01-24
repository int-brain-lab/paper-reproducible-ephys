from figure3_functions import plots_data
from ibllib.atlas import AllenAtlas, Trajectory
from one.api import ONE
from ibllib.pipes.ephys_alignment import EphysAlignment
from reproducible_ephys_functions import labs, figure_style

import numpy as np
import matplotlib.pyplot as plt


one = ONE()
ba = AllenAtlas()

N_REC_PER_LAB = 4
NICKNAMES = False
FIG_SIZE = (6, 4)

lab_number_map, institution_map, lab_colors = labs()
data, lab_colors = plots_data(N_REC_PER_LAB)
data = data.drop_duplicates(subset='subject').reset_index()
data['institution'] = data.lab.map(institution_map)
rec_per_lab = data.groupby('lab_number').size()
data['recording'] = np.concatenate([np.arange(i) for i in rec_per_lab.values])

rows = 3
cols = int(np.ceil(len(data['subject'])/rows))

figure_style()
fig = plt.figure(constrained_layout=False, figsize=FIG_SIZE, dpi=150)
gs = fig.add_gridspec(rows, cols + 1, width_ratios=np.r_[np.ones(cols)*0.9, 0.1])
gs.update(wspace=0.05, hspace=0.3, left=0.1, right=0.9, top=0.9, bottom=0.1)


ml_ranges = []
axs = []
cmin, cmax = np.quantile(ba.image, [0.1, 0.98])

for iR, pid in enumerate(data['pid']):

    print(iR)
    eid, name = one.pid2eid(pid)
    channels = one.load_object(eid, 'channels', collection=f'alf/{name}/pykilosort')
    insertion = one.alyx.rest('insertions', 'list', id=pid)[0]
    traj = one.alyx.rest('trajectories', 'list', provenance='Ephys aligned histology track', probe_insertion=pid)[0]
    feature, track = [*traj['json'][insertion['json']['extended_qc']['alignment_stored']]][:2]
    xyz_picks = np.array(insertion['json']['xyz_picks']) / 1e6
    depths = channels.localCoordinates[:, 1]
    ephysalign = EphysAlignment(xyz_picks, depths, brain_atlas=ba, feature_prev=feature, track_prev=track,
                                speedy=True)
    xyz_chans = channels['mlapdv']
    #xyz_samples = ephysalign.xyz_samples * 1e6
    xyz_samples = xyz_chans
    traj = Trajectory.fit(xyz_samples / 1e6)
    theta = np.arctan(traj.vector[0] / traj.vector[2])
    diff_ml = xyz_chans[:, 0] - traj.point[0] * 1e6
    diff_dv = xyz_chans[:, 2] - traj.point[2] * 1e6
    rotated_ml = (np.cos(theta) * diff_ml - np.sin(theta) * diff_dv) + traj.point[0] * 1e6
    rotated_dv = (np.sin(theta) * diff_ml + np.cos(theta) * diff_dv) + traj.point[2] * 1e6

    LFP_BAND_HIGH = [20, 80]
    lfp = one.load_object(eid, 'ephysSpectralDensityLF', collection=f'raw_ephys_data/{name}')
    freqs = ((lfp['freqs'] > LFP_BAND_HIGH[0])
             & (lfp['freqs'] < LFP_BAND_HIGH[1]))
    chan_power = lfp['power'][:, :]
    lfp_high_region = np.mean(10 * np.log(lfp['power'][freqs]), axis=0)[channels.rawInd]


    vector_perp = np.array([1, -1 * traj.vector[0] / traj.vector[2]])
    extent = 2000
    steps = np.ceil(extent * 2 / ba.res_um).astype(int)
    image = np.zeros((xyz_samples.shape[0], steps))




    ml_range = np.array([0, 0])
    dv_range = np.array([0, 0])
    for i, xyz in enumerate(xyz_samples):
        origin = np.array([xyz[0], xyz[2]])
        anchor = np.r_[[origin + extent * vector_perp], [origin - extent * vector_perp]]
        xargmin = np.argmin(anchor[:, 0])
        xargmax = np.argmax(anchor[:, 0])
        perp_ml = np.linspace(anchor[xargmin, 0], anchor[xargmax, 0], steps)
        perp_ap = np.ones_like(perp_ml) * xyz[1]
        perp_dv = np.linspace(anchor[xargmin, 1], anchor[xargmax, 1], steps)
        if i == 0:
            ml_range = [np.min(perp_ml), np.max(perp_ml)]
            dv_range = [np.min(perp_dv), np.max(perp_dv)]
        else:
            ml_range = [np.min(np.r_[ml_range[0], perp_ml]), np.max(np.r_[ml_range[1], perp_ml])]
            dv_range = [np.min(np.r_[dv_range[0], perp_dv]), np.max(np.r_[dv_range[1], perp_dv])]
        idx = ba.bc.xyz2i(np.c_[perp_ml, perp_ap, perp_dv] / 1e6)
        idx[idx[:, 2] >= ba.image.shape[2], 2] = ba.image.shape[2] - 1
        image[i, :] = ba.image[idx[:, 1], idx[:, 0], idx[:, 2]]
    image = np.flipud(image)
    ml_ranges.append(ml_range)

    ax = fig.add_subplot(gs[int(iR/cols), np.mod(iR, cols)])

    ax.imshow(image, aspect='auto', extent=np.r_[[0, 4000], [0, 3840]], cmap='bone', alpha=1, vmin=cmin, vmax=cmax)
    scat = ax.scatter(channels.localCoordinates[:, 0] + 2000, channels.localCoordinates[:, 1], c=lfp_high_region,
               cmap='viridis', s=2, vmin=-190, vmax=-150)
    axs.append(ax)

ax = fig.add_subplot(gs[0:rows, cols])
cbar = fig.colorbar(scat, cax=ax, ticks=scat.get_clim())
cbar.set_label('Power spectral density', rotation=270, labelpad=-8)
cbar.ax.set_yticklabels([f'{-190} dB', f'{-150} dB'])





for i, subject in enumerate(data['subject']):
    if NICKNAMES:
        axs[i].set_title(subject, rotation=30, ha='left',
                         color=lab_colors[data.loc[i, 'institution']])
    else:
        axs[i].set_title(data.loc[i, 'recording'] + 1,
                         color=lab_colors[data.loc[i, 'institution']])

    if i == 0:
        axs[i].tick_params(axis='y')
        axs[i].spines["right"].set_visible(False)
        axs[i].spines["bottom"].set_visible(False)
        axs[i].spines["top"].set_visible(False)
        axs[i].set_ylabel('Depth along probe (um)')
    else:
        axs[i].set_axis_off()

    axs[i].set(xticks=[])

plt.savefig('figure3_supp2_histology_slices.png')
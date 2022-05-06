from one.api import ONE
import ibllib.atlas as atlas
from ibllib.pipes import histology
import numpy as np
from pathlib import Path
from mayavi import mlab
from atlaselectrophysiology import rendering
from ibllib.plots import color_cycle
import matplotlib.pyplot as plt
from ibllib.ephys.neuropixel import SITES_COORDINATES

one = ONE()
ba = atlas.AllenAtlas(25)

chn_coords = SITES_COORDINATES
depths = chn_coords[:, 1]
MAX_DIST_UM = 500
ACTIVE_LENGTH_UM = np.max(depths).astype(int)
TIP_SIZE_UM = 200


def crawl_up_from_tip(ins, d):
    return (ins.entry - ins.tip) * (d[:, np.newaxis] /
                                    np.linalg.norm(ins.entry - ins.tip)) + ins.tip


def ismember_rows(a, b):
    # Find index of rows a in b
    return np.nonzero(np.all(b == a[:, np.newaxis], axis=2))[1]


traj_rep = one.alyx.rest('trajectories', 'list', provenance='Planned',
                         x=-2243, y=-2000,  project='ibl_neuropixel_brainwide_01')

eid_rep = np.array([traj['session']['id'] for traj in traj_rep])
sess_rep = np.array([traj['session']['subject'] + '_' + traj['session']['start_time'][:10] + '_' +
                     traj['probe_name'] for traj in traj_rep])

traj_rep = traj_rep[0]
ins_repeated = atlas.Insertion.from_dict(traj_rep)

active_sites = (np.array([ACTIVE_LENGTH_UM, 0]) + TIP_SIZE_UM) / 1e6
top_bottom_rep = crawl_up_from_tip(ins_repeated, active_sites)
axis = np.argmax(np.abs(np.diff(top_bottom_rep, axis=0)))

# sample the active track path along this axis
tbi = ba.bc.xyz2i(top_bottom_rep)
nz = tbi[1, axis] - tbi[0, axis] + 1
ishank = np.round(np.array(
    [np.linspace(tbi[0, i], tbi[1, i], nz) for i in np.arange(3)]).T).astype(np.int32)

# Find all indices in column around repeated site recording area
nx = int(
    np.floor(MAX_DIST_UM / 1e6 / np.abs(ba.bc.dxyz[0]) * np.sqrt(2) / 2)) * 2 + 1
ny = int(
    np.floor(MAX_DIST_UM / 1e6 / np.abs(ba.bc.dxyz[1]) * np.sqrt(2) / 2)) * 2 + 1
ixyz = np.stack([v.flatten() for v in np.meshgrid(
    np.arange(-nx, nx + 1), np.arange(-ny, ny + 1), np.arange(nz))]).T
ixyz[:, 0] = ishank[ixyz[:, 2], 0] + ixyz[:, 0]
ixyz[:, 1] = ishank[ixyz[:, 2], 1] + ixyz[:, 1]
ixyz[:, 2] = ishank[ixyz[:, 2], 2]

# Now find the idx of the probes
traj_histology = one.alyx.rest('trajectories', 'list', provenance='Histology track',
                               project='ibl_neuropixel_brainwide_01')
traj_histology = [traj for traj in traj_histology if traj['x'] is not None]
eid = []
chn_roi = []
sess_info = []
for traj_hist in traj_histology:
    insertion = one.alyx.rest('insertions', 'list', session=traj_hist['session']['id'],
                              name=traj_hist['probe_name'])
    xyz_picks = np.array(insertion[0]['json']['xyz_picks'])/1e6
#
    xyz = xyz_picks[np.argsort(xyz_picks[:, 2]), :]
    d = atlas.cart2sph(xyz[:, 0] - xyz[0, 0], xyz[:, 1] - xyz[0, 1], xyz[:, 2] - xyz[0, 2])[0]
    indsort = np.argsort(d)
    xyz = xyz[indsort, :]
    d = d[indsort]
    iduplicates = np.where(np.diff(d) == 0)[0]
    xyz = np.delete(xyz, iduplicates, axis=0)
    #ins_hist = atlas.Insertion.from_dict(traj_hist)
    #top_bottom = crawl_up_from_tip(ins_hist, active_sites)

    chn_xyz = histology.interpolate_along_track(xyz, (depths + TIP_SIZE_UM) / 1e6)

    # mdist = ins_repeated.trajectory.mindist(chn_xyz, bounds=active_sites)
    chn_idx = ba.bc.xyz2i(chn_xyz)

    chn_roi.append(len(ismember_rows(ixyz, chn_idx)))
    eid.append(traj_hist['session']['id'])
    sess_info.append(traj_hist['session']['subject'] + '_' +
                     traj_hist['session']['start_time'][:10] + '_' +
                     traj_hist['probe_name'])


chn_roi = np.array(chn_roi)
eid = np.array(eid)
sess_info = np.array(sess_info)


# Find sessions with > 100 channels in repeated site ROI
idx_chn = np.where(chn_roi > 187)[0]
sess_idx = sess_info[idx_chn]
eid_idx = eid[idx_chn]

is_repeated = np.nonzero(np.isin(sess_idx, sess_rep))[0]
is_not_repeated = np.where(np.isin(sess_idx, sess_rep) == 0)[0]

off_repeated = np.where(np.isin(sess_rep, sess_idx[is_repeated]) == 0)[0]

rep_idx = np.where(np.isin(sess_info, sess_rep) == 1)[0]
off_repeated2 = rep_idx[np.where(np.isin(rep_idx, idx_chn[is_repeated]) == 0)[0]]

ml_is_repeated = np.array([traj_histology[idx_chn[i]]['x'] for i in is_repeated])
ap_is_repeated = np.array([traj_histology[idx_chn[i]]['y'] for i in is_repeated])

ml_is_not_repeated = np.array([traj_histology[idx_chn[i]]['x'] for i in is_not_repeated])
ap_is_not_repeated = np.array([traj_histology[idx_chn[i]]['y'] for i in is_not_repeated])

ml_off_repeated = np.array([traj_histology[i]['x'] for i in off_repeated2])
ap_off_repeated = np.array([traj_histology[i]['y'] for i in off_repeated2])


fig1, ax1 = plt.subplots()
ax1.scatter(traj_rep['x'], traj_rep['y'], color='k', marker="o",
            alpha=0.6)
ax1.scatter(ml_is_not_repeated, ap_is_not_repeated, color=[0.17254902, 0.62745098, 0.17254902],
            marker="o", alpha=0.6)
ax1.scatter(ml_off_repeated, ap_off_repeated, color=[0.83921569, 0.15294118, 0.15686275],
            marker="o", alpha=0.6)
ax1.scatter(ml_is_repeated, ap_is_repeated, color=[1., 0.49803922, 0.05490196], marker="o",
            alpha=0.6)
ax1.scatter(ml_is_not_repeated, ap_is_not_repeated, color=[0.17254902, 0.62745098, 0.17254902],
            marker="o", alpha=0.6)
ax1.scatter(ml_off_repeated, ap_off_repeated, color=[0.83921569, 0.15294118, 0.15686275],
            marker="o", alpha=0.6)

fig = rendering.figure(grid=False)
for iR in is_repeated:
    temp_traj = one.alyx.rest('trajectories', 'list', session=eid_idx[iR],
                              provenance='Histology track', probe=sess_idx[iR][-7:])[0]
    ins = atlas.Insertion.from_dict(temp_traj)
    mlapdv = ba.xyz2ccf(ins.xyz)
    # display the trajectories
    color = color_cycle(1)
    mlab.plot3d(mlapdv[:, 1], mlapdv[:, 2], mlapdv[:, 0],
                line_width=1, tube_radius=20, color=color)

for idx in is_not_repeated:
    temp_traj = one.alyx.rest('trajectories', 'list', session=eid_idx[idx],
                              provenance='Histology track', probe=sess_idx[idx][-7:])[0]
    ins = atlas.Insertion.from_dict(temp_traj)
    mlapdv = ba.xyz2ccf(ins.xyz)
    # display the trajectories
    color = color_cycle(2)
    mlab.plot3d(mlapdv[:, 1], mlapdv[:, 2], mlapdv[:, 0],
                line_width=1, tube_radius=20, color=color)

for idx in off_repeated:
    temp_traj = one.alyx.rest('trajectories', 'list', session=eid_rep[idx],
                              provenance='Histology track', probe=sess_rep[idx][-7:])
    if len(temp_traj) > 0:
        ins = atlas.Insertion.from_dict(temp_traj[0])
        mlapdv = ba.xyz2ccf(ins.xyz)
        # display the trajectories
        color = rendering.color_cycle(4)
        mlab.plot3d(mlapdv[:, 1], mlapdv[:, 2], mlapdv[:, 0],
                    line_width=1, tube_radius=20, color=color)


ins = atlas.Insertion.from_dict(traj_rep)
mlapdv = ba.xyz2ccf(ins.xyz)
color = (0., 0., 0.)
mlab.plot3d(mlapdv[:, 1], mlapdv[:, 2], mlapdv[:, 0],
            line_width=1, tube_radius=40, color=color)

mlab.plot3d(mlapdv[:, 1], mlapdv[:, 2], mlapdv[:, 0],
            line_width=1, tube_radius=250, color=color)

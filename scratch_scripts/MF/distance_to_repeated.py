from one.api import ONE
import ibllib.atlas as atlas
from ibllib.pipes import histology
import numpy as np
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


traj_rep = one.alyx.rest('trajectories', 'list', provenance='Planned',
                         x=-2243, y=-2000,  project='ibl_neuropixel_brainwide_01')
probe_id = [traj['probe_insertion'] for traj in traj_rep]

ins_repeated = atlas.Insertion.from_dict(traj_rep[0])

sites_bounds = crawl_up_from_tip(
    ins_repeated, (np.array([ACTIVE_LENGTH_UM, 0]) + TIP_SIZE_UM) / 1e6)


insertions = one.alyx.rest('insertions', 'list',
                           django='session__projects__name,ibl_neuropixel_brainwide_01,'
                           'json__extended_qc__tracing_exists,True')

distance = []
is_repeated = []
resolved = []
ins_id = []

for ins in insertions:

    xyz_picks = np.array(ins['json']['xyz_picks']) / 1e6
    xyz = xyz_picks[np.argsort(xyz_picks[:, 2]), :]
    d = atlas.cart2sph(xyz[:, 0] - xyz[0, 0], xyz[:, 1] - xyz[0, 1], xyz[:, 2] - xyz[0, 2])[0]
    indsort = np.argsort(d)
    xyz = xyz[indsort, :]
    d = d[indsort]
    iduplicates = np.where(np.diff(d) == 0)[0]
    xyz = np.delete(xyz, iduplicates, axis=0)
    chn_xyz = histology.interpolate_along_track(xyz, (depths + TIP_SIZE_UM) / 1e6)

    sites_bounds = crawl_up_from_tip(
        ins_repeated, (np.array([ACTIVE_LENGTH_UM, 0]) + TIP_SIZE_UM) / 1e6)
    mdist = ins_repeated.trajectory.mindist(chn_xyz, bounds=sites_bounds)
    distance.append(np.mean(mdist))

    resolved.append(ins['json']['extended_qc'].get('alignment_resolved', False))
    ins_id.append(ins['id'])
    if np.isin(ins['id'], probe_id):
        is_repeated.append(True)
    else:
        is_repeated.append(False)


sorted_id = np.argsort(distance)
distance_sorted = np.array(distance)[sorted_id] * 1e6
repeated_sorted = np.array(is_repeated)[sorted_id]
resolved_sorted = np.array(resolved)[sorted_id]
ins_id_sorted = np.array(ins_id)[sorted_id]

color = np.zeros((len(repeated_sorted), 3))
color[repeated_sorted == True] = (1, 0, 0)
color[repeated_sorted == False] = (0, 0, 1)
plt.bar(np.arange(len(distance_sorted)), height=distance_sorted, color=color)
plt.show()

# find insertions that are not resolved and are not repeated site
ins_close_not_resolved = \
    ins_id_sorted[np.bitwise_and(resolved_sorted == False, repeated_sorted == False)]
print(ins_close_not_resolved[0:20])

# This gives me insertions that are not repeated but may or may not be resolved
ins_close = ins_id_sorted[repeated_sorted == False]

ins_rep = ins_id_sorted[repeated_sorted]

fig = rendering.figure(grid=False)
for inse in ins_close_not_resolved[0:20]:

    temp_traj = one.alyx.rest('trajectories', 'list', provenance='Histology track',
                              probe_insertion=inse)[0]

    ins = atlas.Insertion.from_dict(temp_traj)
    mlapdv = ba.xyz2ccf(ins.xyz)
    # display the trajectories
    color = color_cycle(1)
    mlab.plot3d(mlapdv[:, 1], mlapdv[:, 2], mlapdv[:, 0],
                line_width=1, tube_radius=20, color=color)

for inse in ins_rep:

    temp_traj = one.alyx.rest('trajectories', 'list', provenance='Histology track',
                              probe_insertion=inse)[0]

    ins = atlas.Insertion.from_dict(temp_traj)
    mlapdv = ba.xyz2ccf(ins.xyz)
    # display the trajectories
    color = color_cycle(2)
    mlab.plot3d(mlapdv[:, 1], mlapdv[:, 2], mlapdv[:, 0],
                line_width=1, tube_radius=20, color=color)



ins = atlas.Insertion.from_dict(traj_rep[0])
mlapdv = ba.xyz2ccf(ins.xyz)
color = (0., 0., 0.)
mlab.plot3d(mlapdv[:, 1], mlapdv[:, 2], mlapdv[:, 0],
            line_width=1, tube_radius=40, color=color)


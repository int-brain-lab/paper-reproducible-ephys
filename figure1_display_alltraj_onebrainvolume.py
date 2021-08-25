"""
Display all trajectories in one single brain volume
Author: Mayo
"""


from one.api import ONE
import ibllib.atlas as atlas
from mayavi import mlab
from atlaselectrophysiology import rendering
from ibllib.plots import color_cycle


one = ONE()
ba = atlas.AllenAtlas(25)
traj_rep = one.alyx.rest('trajectories', 'list', provenance='Planned',
                         x=-2243, y=-2000,  project='ibl_neuropixel_brainwide_01')


fig = rendering.figure(grid=False)
for traj in traj_rep:
    temp_traj = one.alyx.rest('trajectories', 'list',
                                 provenance='Ephys aligned histology track',
                                 probe_insertion=traj['probe_insertion'])
    if len(temp_traj) == 0:
        temp_traj = one.alyx.rest('trajectories', 'list', provenance='Histology track',
                                  probe_insertion=traj['probe_insertion'])

        if len(temp_traj) == 0:
            continue

    if not temp_traj[0]['x']:
        continue

    ins = atlas.Insertion.from_dict(temp_traj[0])
    mlapdv = ba.xyz2ccf(ins.xyz)
    # display the trajectories
    color = color_cycle(1)
    mlab.plot3d(mlapdv[:, 1], mlapdv[:, 2], mlapdv[:, 0],
                line_width=1, tube_radius=20, color=color)

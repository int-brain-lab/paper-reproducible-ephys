"""
Display all trajectories in one single brain volume
Author: Mayo, Gaelle
"""

import numpy as np
from iblatlas.regions import BrainRegions
from one.api import ONE
import iblatlas.atlas as atlas
from mayavi import mlab
from atlaselectrophysiology import rendering
from ibllib.plots import color_cycle


one = ONE()
ba = atlas.AllenAtlas(25)
traj_rep = one.alyx.rest('trajectories', 'list', provenance='Planned',
                         x=-2243, y=-2000,  project='ibl_neuropixel_brainwide_01', use_cache=False)

# TODO removing PID manually for sake of figure, but need to iron this out in Alyx
except_pid = ['8b735d77-b77b-4243-8821-37802bf402fe',
              '94af9073-0914-4323-a90a-5eea1ef5f92c']

fig = rendering.figure(grid=False)
for traj in traj_rep:

    if traj['probe_insertion'] not in except_pid:
        temp_traj = one.alyx.rest('trajectories', 'list',
                                     provenance='Ephys aligned histology track',
                                     probe_insertion=traj['probe_insertion'], use_cache=False)
        if len(temp_traj) == 0:
            temp_traj = one.alyx.rest('trajectories', 'list', provenance='Histology track',
                                      probe_insertion=traj['probe_insertion'], use_cache=False)

            if len(temp_traj) == 0:
                continue

        if not temp_traj[0]['x']:
            continue

        ins = atlas.Insertion.from_dict(temp_traj[0])
        mlapdv = ba.xyz2ccf(ins.xyz)
        # display the trajectories
        color = (0., 0., 0.)  # color_cycle(1)
        mlab.plot3d(mlapdv[:, 1], mlapdv[:, 2], mlapdv[:, 0],
                    line_width=2, tube_radius=20, color=color)

        # # Display pid on top of it, for debugging purpose
        # label = traj['probe_insertion']
        # mlab.text3d(mlapdv[0, 1], mlapdv[0, 2], mlapdv[0, 0] - 500, label,
        #             line_width=4, color=tuple(color), figure=fig, scale=150)


'''
Display structure meshes within the brain volume
You can download the mesh object for each brain structure here:
http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/structure_meshes/
'''
br = BrainRegions()

target_area = ['VISa', 'CA1', 'DG', 'LP', 'PO']
for target in target_area:
    rindx_void = np.where(br.acronym == target)
    idx = rindx_void[0][0]
    mesh_id = br.id[idx]
    # print(mesh_id) --> useful to download the specific mesh obj from the Allen website
    color = br.rgb[idx, :]/255
    path = f'/Users/gaelle/Desktop/Allenmesh/{mesh_id}.obj.txt'
    rendering.add_mesh(fig, path, color, opacity=0.6)

'''
Supplementary Fig 1-1a
Plot selected trajectories, with specific brain regions highlighted
'''
# Author: Gaelle

import numpy as np
from iblatlas.regions import BrainRegions
from one.api import ONE
import iblatlas.atlas as atlas
from mayavi import mlab
from atlaselectrophysiology import rendering
from ibllib.plots import color_cycle

one = ONE()
ba = atlas.AllenAtlas(25)

# Selected list of traj to display
list_pid1 = ['4b93a168-0f3b-4124-88fa-a57046ca70e1',  # BAD - Low yield
             '57656bee-e32e-4848-b924-0f6f18cfdfb1',
             'c4f6665f-8be5-476b-a6e8-d81eeae9279d']

list_pid2 = ['523f8301-4f56-4faf-ab33-a9ff11331118',  # GOOD
             '63517fd4-ece1-49eb-9259-371dc30b1dd6',
             'a12c8ae8-d5ad-4d15-b805-436ad23e5ad1',
             'c07d13ed-e387-4457-8e33-1d16aed3fd92',
             'eeb27b45-5b85-4e5c-b6ff-f639ca5687de',
             # 'f86e9571-63ff-4116-9c40-aa44d57d2da9',
             'f936a701-5f8a-4aa1-b7a9-9f8b5b69bc7c']

list_pid3 = ['7cbecb3f-6a8a-48e5-a3be-8f7a762b5a04',  # BAD - Miss target
            '8ca1a850-26ef-42be-8b28-c2e2d12f06d6',
            '63a32e5c-f63a-450d-85cb-140947b67eaf']

list_pid_overall = [list_pid1, list_pid2, list_pid3]

color_list_overall = [(1, 0, 0), color_cycle(1), (0, 0, 1)]

# fig = rendering.figure(grid=False)
fig = mlab.figure(bgcolor=(1, 1, 1))

for list_pid, color_list in zip(list_pid_overall, color_list_overall):
    for pid in list_pid:
        temp_traj = one.alyx.rest('trajectories', 'list', probe_insertion=pid,
                                  provenance='Ephys aligned histology track')
        if len(temp_traj) == 0:
            temp_traj = one.alyx.rest('trajectories', 'list', probe_insertion=pid,
                                      provenance='Histology track')
            if len(temp_traj) == 0:
                continue

        if not temp_traj[0]['x']:
            continue

        ins = atlas.Insertion.from_dict(temp_traj[0])
        mlapdv = ba.xyz2ccf(ins.xyz)
        # display the trajectories
        color = color_cycle(1)
        mlab.plot3d(mlapdv[:, 1], mlapdv[:, 2], mlapdv[:, 0],
                    line_width=1, tube_radius=20, color=color_list)

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
    color = br.rgb[idx,:]/255
    path = f'/Users/gaelle/Desktop/Allenmesh/{mesh_id}.obj.txt'
    rendering.add_mesh(fig, path, color, opacity=0.6)

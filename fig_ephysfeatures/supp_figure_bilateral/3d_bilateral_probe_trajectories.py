#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 13:36:54 2022
By: Steven West & Guido Meijer
"""

from pathlib import Path

from one.api import ONE
import iblatlas.atlas as atlas

from mayavi import mlab
from atlaselectrophysiology import rendering
from reproducible_ephys_functions import LAB_MAP


lab_number_map, institution_map, lab_colors = LAB_MAP()

eids = ['0802ced5-33a3-405e-8336-b65ebc5cb07c',
        '7cb81727-2097-4b52-b480-c89867b5b34c',
        '7f6b86f9-879a-4ea2-8531-294a221af5d0',
        'dda5fc59-f09a-4256-9fb5-66c67667a466',
        'ecb5520d-1358-434c-95ec-93687ecd1396',
        'f312aaec-3b6f-44b3-86b4-3a0c119c0438']


one = ONE()
ba = atlas.AllenAtlas(25)

fig = rendering.figure(grid=False)

for i,e in enumerate(eids):

    # Get lab color
    ses_details = one.get_details(e)
    color = lab_colors[institution_map[ses_details['lab']]]

    trajs = one.alyx.rest('trajectories', 'list', session=e, provenance='Histology track')

    for t in trajs:

        phi_eid = t['session']['id']
        phi_probe = t['probe_name']
        phi_subj = t['session']['subject']
        phi_lab = t['session']['lab']

        phi_traj = one.alyx.rest('trajectories', 'list', session=phi_eid,
                                 provenance='Histology track', probe=phi_probe)[0]
        ins = atlas.Insertion.from_dict(phi_traj)

        mlapdv = ba.xyz2ccf(ins.xyz)
        # display the trajectories
        mlab.plot3d(mlapdv[:, 1], mlapdv[:, 2], mlapdv[:, 0],
                    line_width=1, tube_radius=40, color=color)


rendering.rotating_video( str(Path('guido_eids_3D.webm')), fig)
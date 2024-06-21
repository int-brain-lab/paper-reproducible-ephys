"""
For a single subject:
3D rendering trajectories - Accentuate on repeated site
Plot raster + drift for rep site
===========================
Generates 3D rendering of all probe trajectories for a single subject.
Repeated site plotted in red, other sites in black.
"""
# Author: Gaelle
# inspired from:
# https://github.com/int-brain-lab/ibllib/blob/master/examples/one/histology/visualization3D_alyx_traj_planned_histology.py
# Could diplay channels instead of traj, cf:
# https://github.com/int-brain-lab/ibllib/blob/master/examples/one/histology/docs_visualization3D_subject_channels.py

from mayavi import mlab
from atlaselectrophysiology import rendering
import iblatlas.atlas as atlas
from oneibl.one import ONE
import alf.io
from brainbox.metrics import electrode_drift
import matplotlib.pyplot as plt
import brainbox.plot as bbplot

one = ONE(base_url="https://alyx.internationalbrainlab.org")
subject = 'CSHL045'

ba = atlas.AllenAtlas(25)

str_query = 'probe_insertion__session__projects__name__icontains,ibl_neuropixel_brainwide_01,' \
            'probe_insertion__session__qc__lt,50,' \
            'probe_insertion__session__extended_qc__behavior,1'


def plot_traj(traj, ba=atlas.AllenAtlas(25), fig_handle=None,
              line_width=3, color=(0., 0., 0.), tube_radius=20,
              label_text=None, label_width=4, label_color=(0., 0., 0.), label_shift=500):

    ins = atlas.Insertion.from_dict(traj)
    mlapdv = ba.xyz2ccf(ins.xyz)

    if fig_handle is None:
        # Plot empty atlas template
        fig_handle = rendering.figure()

    mlab.plot3d(mlapdv[:, 1], mlapdv[:, 2], mlapdv[:, 0],
                line_width=line_width, color=color, tube_radius=tube_radius, figure=fig_handle)
    if label_text is not None:
        # Setup the  label at the top of the trajectory
        mlab.text3d(mlapdv[0, 1], mlapdv[0, 2], mlapdv[0, 0] - label_shift, label_text,
                    line_width=label_width, color=label_color, figure=fig_handle, scale=500)

    return fig_handle

# Get the aligned + agreed trajectories (and micro manip in case alignemnt not done / resolved)
traj_sub_aligned = one.alyx.rest('trajectories', 'list', provenance='Ephys aligned histology track',
                                 subject=subject,
                                 django=str_query+',probe_insertion__json__extended_qc__alignment_resolved,True')
ins_id_align = [item['probe_insertion'] for item in traj_sub_aligned]

traj_sub_all = traj_rep = one.alyx.rest('trajectories', 'list', provenance='Micro-manipulator',
                                        subject=subject,
                                        django=str_query)

# One figure for all
fig_handle = rendering.figure()

# Display the repeated site track in red
traj_rep = one.alyx.rest('trajectories', 'list', provenance='Planned',
                         x=-2243, y=-2000,
                         subject=subject,
                         django=str_query)
ins_id_rep = traj_rep[0]['probe_insertion']
traj_rep_align = [item for item in traj_sub_aligned if item['probe_insertion'] in ins_id_rep]
traj_rep_align=traj_rep_align[0]

if len(traj_rep_align) > 0:
    color = (1., 0., 0.)
    plot_traj(traj=traj_rep_align, fig_handle=fig_handle,
              color=color, label_text='Repeated site', label_color=color)

# Display other traj in black
for i_traj in range(0, len(traj_sub_all)):
    traj_mic = traj_sub_all[i_traj]

    if traj_mic['probe_insertion'] not in ins_id_rep:
        if traj_mic['probe_insertion'] in ins_id_align:
            traj = [item for item in traj_sub_aligned if item['probe_insertion'] in traj_mic['probe_insertion']]
            traj = traj[0]
        else:
            print('No resolved alignment found, using micromanip instead')
            traj = traj_mic
        plot_traj(traj=traj, fig_handle=fig_handle)

# Plot raster + drift
# inspired from docs_compute_drift.py

# Specify subject, date and probe we are interested in
date = traj_rep_align['session']['start_time'][0:10]
sess_no = traj_rep_align['session']['number']
probe_label = traj_rep_align['probe_name']
eid = one.search(subject=subject, date=date, number=sess_no)[0]

# define datasets to download
dtypes = ['spikes.times',
          'spikes.depths',
          'spikes.amps']

# Download the data and get paths to downloaded data
_ = one.load(eid, dataset_types=dtypes, download_only=True)
alf_path = one.path_from_eid(eid).joinpath('alf', probe_label)

# Load in spikes object and use brainbox function to compute drift over session
spikes = alf.io.load_object(alf_path, 'spikes')
drift = electrode_drift.estimate_drift(spikes['times'], spikes['amps'], spikes['depths'],
                                       display=True)

# bbplot.driftmap(spikes['times'],
#                 spikes['depths'],
#                 plot_style='bincount')

plt.show()

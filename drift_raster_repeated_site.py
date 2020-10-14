"""
Compute drift for example sessions using: (REPEATED SITE)
https://github.com/int-brain-lab/ibllib/blob/master/brainbox/metrics/electrode_drift.py
and display raster plot below
"""
# Authors: Gaelle, Olivier

# from reproducible_ephys_paths import FIG_PATH

from brainbox.metrics.electrode_drift import estimate_drift
from oneibl.one import ONE
import brainbox.plot as bbplot
import matplotlib.pyplot as plt
import numpy as np
import alf.io
# from pathlib import Path

one = ONE()

# Find sessions
dataset_types = ['spikes.times',
                 'spikes.amps',
                 'spikes.depths']

traj = one.alyx.rest('trajectories', 'list', provenance='Planned',
                     x=-2243, y=-2000,  # repeated site coordinate
                     project='ibl_neuropixel_brainwide_01',
                     django='probe_insertion__session__qc__lt,50')  # All except CRITICAL
eids_traj = np.array([p['session']['id'] for p in traj])
probe_name = np.array([t['probe_name'] for t in traj])

eids_ds = one.search(dataset_types=dataset_types,
                     project='ibl_neuropixel_brainwide_01',
                     task_protocol='_iblrig_tasks_ephysChoiceWorld')

indx = np.where(np.isin(eids_traj, eids_ds))[0]
eids = eids_traj[indx]
probes = probe_name[indx]

# fig_path = Path(FIG_PATH)
fig_path = '/Users/gaelle/Desktop/Drift_RS/'

for i_eid in range(0, len(eids)):
    eid = eids[i_eid]
    session_path = one.path_from_eid(eid)
    probe_path = session_path.joinpath('alf', probes[i_eid])
    # Get dataset
    _ = one.load(eid, dataset_types, download_only=True)
    spikes = alf.io.load_object(probe_path, 'spikes')
    drift = estimate_drift(spikes['times'], spikes['amps'], spikes['depths'], display=False)

    # save data into Flatiron local download directory
    np.save(probe_path.joinpath('drift.npy'), drift)

    # PLOT
    # Tight layout
    fig3 = plt.figure(constrained_layout=True)
    gs = fig3.add_gridspec(3, 3)
    f3_ax0 = fig3.add_subplot(gs[0, :])
    f3_ax0.plot(drift)
    f3_ax1 = fig3.add_subplot(gs[1:, :])
    bbplot.driftmap(spikes['times'],
                    spikes['depths'],
                    ax=f3_ax1, plot_style='bincount')
    f3_ax0.set_xlim(f3_ax1.get_xlim())

    # Save plot
    fname = f'{fig_path}Drift__{eid}'
    plt.savefig(fname)
    plt.close()

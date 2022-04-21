from oneibl.one import ONE
import numpy as np
from ibllib.ephys.neuropixel import SITES_COORDINATES
from ibllib.pipes.ephys_alignment import EphysAlignment
from ibllib.atlas import atlas


def sort_repeated_site_by_distance(subjects, dates, probes, reference='repeated', one=None,
                                   brain_atlas=None):
    one = one or ONE()
    brain_atlas = brain_atlas or atlas.AllenAtlas(25)
    depths = SITES_COORDINATES[:, 1]

    entry = []
    tip = []
    channels = []
    for subj, date, probe_label in zip(subjects, dates, probes):
        eid = one.search(subject=subj, date=date)[0]
        insertion = one.alyx.rest('insertions', 'list', session=eid, name=probe_label)
        xyz_picks = np.array(insertion[0].get('json').get('xyz_picks', 0)) / 1e6
        trajectory = one.alyx.rest('trajectories', 'list', provenance='Histology track',
                                   probe_insertion=insertion[0]['id'])

        ins = atlas.Insertion.from_dict(trajectory[0])
        ephysalign = EphysAlignment(xyz_picks, depths, brain_atlas=brain_atlas)
        xyz_channels = ephysalign.get_channel_locations(ephysalign.feature_init,
                                                        ephysalign.track_init)

        entry.append(ins.entry)
        tip.append(ins.tip)
        channels.append(xyz_channels)

    min_dist = []
    if reference == 'repeated':
        traj_rep = one.alyx.rest('trajectories', 'list', provenance='Planned',
                                 x=-2243, y=-2000, project='ibl_neuropixel_brainwide_01')
        traj_rep = traj_rep[0]
        ins_repeated = atlas.Insertion.from_dict(traj_rep)
        for chans in channels:
            dist = ins_repeated.trajectory.mindist(chans)
            min_dist.append(np.mean(dist))

    elif reference == 'mean':
        xyz = np.vstack([np.mean(entry, axis=0), np.mean(tip, axis=0)])
        traj_repeated = atlas.Trajectory.fit(xyz)
        for chans in channels:
            dist = traj_repeated.mindist(chans)
            min_dist.append(np.mean(dist))

    elif reference == 'median':
        xyz = np.vstack([np.median(entry, axis=0), np.median(tip, axis=0)])
        traj_repeated = atlas.Trajectory.fit(xyz)
        for chans in channels:
            dist = traj_repeated.mindist(chans)
            min_dist.append(np.mean(dist))

    distance = np.sort(min_dist)
    dist_idx = np.argsort(min_dist)

    return distance, dist_idx



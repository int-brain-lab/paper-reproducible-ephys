"""
Compare clusters across alignments
=============================
Extract locations of clusters from different user alignments for same session
Create plots comparing the number of clusters in different brain regions across alignments and
the difference in the average euclidean distance between clusters for different alignments
"""

# import modules
from oneibl.one import ONE
from ibllib.pipes.ephys_alignment import EphysAlignment
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ibllib.atlas as atlas
import pandas as pd
from pathlib import Path

# Instantiate brain atlas and one
brain_atlas = atlas.AllenAtlas(25)
one = ONE()
fig_path = Path('/home/guido/Figures/Ephys/RepeatedSite')

# Get all repeated sites and the associated probe insertions
traj = one.alyx.rest('trajectories', 'list', provenance='Planned',
                     x=-2243, y=-2000,  # repeated site coordinate
                     project='ibl_neuropixel_brainwide_01',
                     django='probe_insertion__session__qc__lt,50')  # All except CRITICAL

probe_insertion = [t['probe_insertion'] for t in traj]

for p in probe_insertion:
    trajectory = one.alyx.rest('trajectories', 'list', provenance='Ephys aligned histology track',
                               probe_insertion=p)
    if len(trajectory) != 0 and len(trajectory[0]['json']) > 1:

        eid = trajectory[0]['session']['id']
        probe_label = trajectory[0]['probe_name']
        subject = trajectory[0]['session']['subject']
        date = trajectory[0]['session']['start_time'][0:10]

        # Load in channels.localCoordinates dataset type
        chn_coords = one.load(eid, dataset_types=['channels.localCoordinates'])[0]
        depths = chn_coords[:, 1]

        # Load in the clusters.channels
        _ = one.load(eid, dataset_types=['clusters.channels'])
        alf_path = one.path_from_eid(eid).joinpath('alf', probe_label)
        cluster_chans = np.load(alf_path.joinpath('clusters.channels.npy'))

        # Find the ephys aligned trajectory for eid probe combination
        trajectory = one.alyx.rest('trajectories', 'list',
                                   provenance='Ephys aligned histology track',
                                   session=eid, probe=probe_label)
        # Extract all alignments from the json field of object
        alignments = trajectory[0]['json']

        # Load in the initial user xyz_picks obtained from track tracing
        insertion = one.alyx.rest('insertions', 'list', session=eid, name=probe_label)
        xyz_picks = np.array(insertion[0]['json']['xyz_picks']) / 1e6

        # Iterate over all alignments for trajectory and store output in clusters dict also create
        # pd dataframe with info about no. of clusters in each region for each alignment
        clusters = dict()

        for iK, key in enumerate(alignments):

            # Location of reference lines used for alignment
            feature = np.array(alignments[key][0])
            track = np.array(alignments[key][1])

            # Instantiate EphysAlignment object
            ephysalign = EphysAlignment(xyz_picks, depths, track_prev=track, feature_prev=feature,
                                        brain_atlas=brain_atlas)

            # Find xyz location of all channels
            xyz_channels = ephysalign.get_channel_locations(feature, track)
            # Find brain region that each channel is located in
            brain_regions = ephysalign.get_brain_locations(xyz_channels)

            # Find the location of clusters along the alignment - both in acronym and in xyz
            cluster_info = dict()
            cluster_info['regions'] = brain_regions['acronym'][cluster_chans]
            cluster_info['xyz'] = xyz_channels[cluster_chans, :]

            # Store cluster info in channels dict with same key as alignment
            clusters.update({key: cluster_info})

            # Find number of clusters in each brain region and store in pd dataframe
            reg, n_reg = np.unique(clusters[key]['regions'], return_counts=True)

            temp_align = pd.DataFrame({'Brain Region': reg, 'User': [key] * len(reg),
                                       'Count': n_reg})
            if iK == 0:
                session_summary = temp_align
            else:
                session_summary = pd.concat([session_summary, temp_align])

        # Plot the number of clusters in each brain region for each alignment
        # TODO display in order of regions in the brain
        # Complicated there are for example 2 separate DG-mo regions along track - do we want to
        # distinguish?? (another day's problem!)
        session_summary = session_summary.pivot('Brain Region', 'User', 'Count')
        session_summary = session_summary.fillna(0)
        x_labels = [col[20:] for col in session_summary.columns]
        fig, ax = plt.subplots(1, figsize=(10, 10))
        sns.heatmap(session_summary, annot=True, fmt="0.0f", xticklabels=x_labels, cmap='magma',
                    cbar_kws={'label': 'no. of clusters'}, ax=ax)
        fig.suptitle(subject + '_' + str(date) + '_' + probe_label, fontsize=14)
        plt.show()
        fig.savefig(fig_path.joinpath('n_clust_' + subject + '_' + str(date) + '_' + probe_label
                                      + '.png'), dpi=600)
        plt.close(fig)

        # Compute average euclidean distance in 3D space between clusters
        cluster_dist = np.zeros((len(clusters), len(clusters)))
        user = []
        for ik, key in enumerate(clusters):
            for ikk, key2 in enumerate(clusters):
                cluster_dist[ik, ikk] = np.mean(np.sqrt(np.sum((clusters[key]['xyz'] -
                                                                clusters[key2]['xyz']) ** 2,
                                                               axis=1)), axis=0) * 1e6
            user.append(key[20:])

        mask = np.zeros_like(cluster_dist)
        mask[np.triu_indices_from(mask)] = True
        fig2, ax2 = plt.subplots(1, figsize=(10, 10))
        sns.heatmap(cluster_dist, annot=True, fmt="0.2f", xticklabels=user, yticklabels=user,
                    cmap='cividis',cbar_kws={'label': 'average distance between clusters (um)'},
                    ax=ax2)
        fig2.suptitle(subject + '_' + str(date) + '_' + probe_label, fontsize=14)
        plt.show()
        fig2.savefig(fig_path.joinpath('avg_dist_' + subject + '_' + str(date) + '_' + probe_label
                                       + '.png'), dpi=600)
        plt.close(fig2)


"""
Compare clusters across alignments
=============================
Extract locations of clusters from different user alignments for same session
Create plots comparing the number of clusters in different brain regions across alignments and
the difference in the average euclidean distance between clusters for different alignments
"""

# import modules
from oneibl.one import ONE
from reproducible_ephys_paths import FIG_PATH
from ibllib.pipes.ephys_alignment import EphysAlignment
from ibllib.ephys.neuropixel import SITES_COORDINATES
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from iblatlas import atlas
from iblatlas.regions import import regions_from_allen_csv
import pandas as pd
from pathlib import Path

# Instantiate brain atlas and one
brain_atlas = atlas.AllenAtlas(25)
one = ONE()
fig_path = Path(FIG_PATH)

r = regions_from_allen_csv()


def plot_regions(region, label, colour, ax):

    for reg, col in zip(region, colour):
        height = np.abs(reg[1] - reg[0])
        bottom = reg[0]
        color = col / 255
        ax.bar(x=0.7, height=height, width=1, color=color, bottom=reg[0], edgecolor='w')
    sec_ax = ax.secondary_yaxis('right')
    sec_ax.set_yticks(label[:, 0].astype(int))
    sec_ax.set_yticklabels(label[:, 1])
    sec_ax.yaxis.set_tick_params(labelsize=10)
    sec_ax.tick_params(axis="y", direction="in", pad=-100)
    sec_ax.set_ylim([20, 3840])
    sec_ax.set_zorder(50)
    ax.get_xaxis().set_visible(False)

def plot_scaling(region, scale, mapper, ax):
    for reg, col in zip(region_scaled, scale_factor):
        height = np.abs(reg[1] - reg[0])
        bottom = reg[0]
        color = np.array(mapper.to_rgba(col, bytes=True)) / 255
        ax.bar(x=0, height=height, width=0.2, color=color, bottom=reg[0], edgecolor='w')

    ax.set_yticks(np.mean(region, axis=1))
    ax.set_yticklabels(np.around(scale, 2))
    ax.tick_params(axis="y", direction="in")
    ax.set_ylim([20, 3840])


norm = matplotlib.colors.Normalize(vmin=0.5, vmax=1.5, clip=True)
mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.seismic)

# Get all repeated sites and the associated probe insertions
#traj = one.alyx.rest('trajectories', 'list', provenance='Planned',
#                     x=-2243, y=-2000,  # repeated site coordinate
#                     project='ibl_neuropixel_brainwide_01',
#                     django='probe_insertion__session__qc__lt,50')  # All except CRITICAL
traj = one.alyx.rest('trajectories', 'list', provenance='Planned',
                     project='ibl_neuropixel_brainwide_01')

probe_insertion = [t['probe_insertion'] for t in traj]
depths = SITES_COORDINATES[:, 1]

for p in probe_insertion:
    trajectory = one.alyx.rest('trajectories', 'list', provenance='Ephys aligned histology track',
                               probe_insertion=p)
    if len(trajectory) != 0 and len(trajectory[0]['json']) > 1:

        # Get rid of multiple alignments by same person.
        alignments = trajectory[0]['json']
        sorted_alignments = [*alignments.keys()]
        sorted_alignments = sorted(sorted_alignments, reverse=True)
        sorted_names = [sort[20:] for sort in sorted_alignments]
        _, keep_keys = np.unique(np.array(sorted_names), return_index=True)
        rm_key = np.setdiff1d(np.arange(len(sorted_names)), keep_keys)
        for rm in rm_key:
            _ = alignments.pop(sorted_alignments[rm])

        if len(alignments) > 1:
            eid = trajectory[0]['session']['id']
            probe_label = trajectory[0]['probe_name']
            subject = trajectory[0]['session']['subject']
            date = trajectory[0]['session']['start_time'][0:10]

            # Load in the clusters.channels
            _ = one.load(eid, dataset_types=['clusters.channels'])
            alf_path = one.path_from_eid(eid).joinpath('alf', probe_label)
            cluster_chans = np.load(alf_path.joinpath('clusters.channels.npy'))

            # Load in the initial user xyz_picks obtained from track tracing
            insertion = one.alyx.rest('insertions', 'list', session=eid, name=probe_label)
            xyz_picks = np.array(insertion[0]['json']['xyz_picks']) / 1e6

            # Iterate over all alignments for trajectory and store output in clusters dict also
            # create pd dataframe with info about no. of clusters in each region for each alignment
            clusters = dict()
            channels = dict()

            fig = plt.figure(figsize=(15, 15))
            if len(alignments) < 3:
                gs = fig.add_gridspec(12, len(alignments) + 1)
            else:
                gs = fig.add_gridspec(12, len(alignments) + 2)

            for iK, key in enumerate(alignments):

                # Location of reference lines used for alignment
                feature = np.array(alignments[key][0])
                track = np.array(alignments[key][1])

                # Instantiate EphysAlignment object
                ephysalign = EphysAlignment(xyz_picks, depths, track_prev=track,
                                            feature_prev=feature,
                                            brain_atlas=brain_atlas)

                # Find xyz location of all channels
                xyz_channels = ephysalign.get_channel_locations(feature, track)
                # Find brain region that each channel is located in
                brain_regions = ephysalign.get_brain_locations(xyz_channels)
                region, region_label = ephysalign.scale_histology_regions(feature, track)
                region_colour = ephysalign.region_colour
                region_scaled, scale_factor = ephysalign.get_scale_factor(region)

                # Find the location of clusters along the alignment - both in acronym and in xyz
                cluster_info = dict()
                cluster_info['regions'] = brain_regions['acronym'][cluster_chans]
                cluster_info['brain_id'] = brain_regions['id'][cluster_chans]
                cluster_info['parent_id'] = r.get(ids=cluster_info['brain_id']).parent.astype(int)
                cluster_info['xyz'] = xyz_channels[cluster_chans, :]

                channel_info = dict()
                channel_info['regions'] = brain_regions['acronym']
                channel_info['brain_id'] = brain_regions['id']
                channel_info['parent_id'] = r.get(ids=channel_info['brain_id']).parent.astype(int)
                channel_info['xyz'] = xyz_channels

                # Store cluster info in channels dict with same key as alignment
                clusters.update({key: cluster_info})
                channels.update({key: channel_info})

                # Find number of clusters in each brain region and store in pd dataframe
                reg, n_reg = np.unique(clusters[key]['regions'], return_counts=True)

                temp_align = pd.DataFrame({'Brain Region': reg, 'User': [key] * len(reg),
                                           'Count': n_reg})
                if iK == 0:
                    session_summary = temp_align
                else:
                    session_summary = pd.concat([session_summary, temp_align])

                ax_i = fig.add_subplot(gs[:, iK])
                plot_scaling(region_scaled, scale_factor, mapper, ax_i)
                plot_regions(region, region_label, region_colour, ax_i)
                ax_i.set_title(key[20:])

            # Plot the number of clusters in each brain region for each alignment
            session_summary = session_summary.pivot('Brain Region', 'User', 'Count')
            session_summary = session_summary.fillna(0)
            # x_labels = [col[20:25] for col in session_summary.columns]

            # fig, ax = plt.subplots(1, figsize=(10, 10))
            # sns.heatmap(session_summary, annot=True, fmt="0.0f", xticklabels=x_labels,
            #             cmap='magma', cbar_kws={'label': 'no. of clusters'}, ax=ax)
            # fig.suptitle(subject + '_' + str(date) + '_' + probe_label, fontsize=14)
            # plt.show()
            # fig.savefig(fig_path.joinpath('n_clust_' + subject + '_' + str(date) + '_' +
            #                               probe_label + '.png'), dpi=600)
            # plt.close(fig)

            # Compute the sum of abs difference in clusters between same brain regions
            diff = np.zeros((len(clusters), len(clusters)))
            user = []
            for ic, col in enumerate(clusters):
                for icc, col2 in enumerate(clusters):
                    diff[ic, icc] = np.sum(np.abs(np.diff(np.c_[session_summary[col].values,
                                                                session_summary[col2].values],
                                                          axis=1)))
                user.append(col[20:25])

            # Normalise according to total number of clusters
            diff_norm = diff / np.sum(session_summary[col].values)
            if len(alignments) < 3:
                ax_i = fig.add_subplot(gs[0:3, len(alignments)])
            else:
                ax_i = fig.add_subplot(gs[0:3, len(alignments):len(alignments) + 2])
            sns.heatmap(diff_norm, annot=True, fmt="0.2f", xticklabels=user,
                        yticklabels=user, cmap='cividis', vmax=1, vmin=0, ax=ax_i)
            ax_i.set_title('difference')

            # Compute average euclidean distance in 3D space between clusters
            cluster_dist = np.zeros((len(clusters), len(clusters)))
            for ik, key in enumerate(clusters):
                for ikk, key2 in enumerate(clusters):
                    cluster_dist[ik, ikk] = np.mean(np.sqrt(np.sum((clusters[key]['xyz'] -
                                                                    clusters[key2]['xyz']) ** 2,
                                                                   axis=1)), axis=0) * 1e6
            if len(alignments) < 3:
                ax_i = fig.add_subplot(gs[4:7, len(alignments)])
            else:
                ax_i = fig.add_subplot(gs[4:7, len(alignments):len(alignments) + 2])

            sns.heatmap(cluster_dist, annot=True, fmt="0.2f", xticklabels=user, yticklabels=user,
                        cmap='cividis', vmax=300, vmin=0, ax=ax_i)
            ax_i.set_title('distance (um)')

            cluster_sim = np.zeros((len(clusters), len(clusters)))
            for ik, key in enumerate(clusters):
                for ikk, key2 in enumerate(clusters):
                    same_id = np.where(clusters[key]['brain_id'] == clusters[key2]['brain_id'])[0]
                    not_same_id = \
                    np.where(clusters[key]['brain_id'] != clusters[key2]['brain_id'])[0]
                    same_parent = np.where(clusters[key]['parent_id'][not_same_id] ==
                                           clusters[key2]['parent_id'][not_same_id])[0]
                    cluster_sim[ik, ikk] = len(same_id) + (len(same_parent) * 0.5)

            # Normalise
            cluster_sim_norm = cluster_sim / np.max(cluster_sim)
            if len(alignments) < 3:
                ax_i = fig.add_subplot(gs[8:11, len(alignments)])
            else:
                ax_i = fig.add_subplot(gs[8:11, len(alignments):len(alignments) + 3])

            sns.heatmap(cluster_sim_norm, annot=True, fmt="0.2f", xticklabels=user,
                        yticklabels=user, cmap='cividis', vmax=1, vmin=0,
                        ax=ax_i)
            ax_i.set_title('similarity')

            fig.suptitle(subject + '_' + str(date) + '_' + probe_label, fontsize=14)
            plt.show()
            fig.savefig(fig_path.joinpath('alignment_' + subject + '_' + str(date) + '_' +
                                          probe_label + '.png'), dpi=600)
            plt.close(fig)
        else:
            continue


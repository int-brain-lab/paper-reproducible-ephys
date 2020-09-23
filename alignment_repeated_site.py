"""

Plot alignment for repeated site recordings
Code inspired from iblapps > atlaselectrophysiology > compare_alignments

"""
# author : Gaelle, Mayo


# import modules
from oneibl.one import ONE
from ibllib.pipes.ephys_alignment import EphysAlignment
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import ibllib.atlas as atlas
from pathlib import Path

# Functions

def plot_regions(region, label, colour, ax):
    for reg, col in zip(region, colour):
        height = np.abs(reg[1] - reg[0])
        bottom = reg[0]
        color = col / 255
        ax.bar(x=0.5, height=height, width=1, color=color, bottom=reg[0], edgecolor='w')

    ax.set_yticks(label[:, 0].astype(int))
    ax.set_yticklabels(label[:, 1])
    ax.yaxis.set_tick_params(labelsize=10)
    ax.tick_params(axis="y", direction="in", pad=-50)
    ax.set_ylim([20, 3840])
    ax.get_xaxis().set_visible(False)


def plot_scaling(region, scale, mapper, ax):
    for reg, col in zip(region_scaled, scale_factor):
        height = np.abs(reg[1] - reg[0])
        bottom = reg[0]
        color = np.array(mapper.to_rgba(col, bytes=True)) / 255
        ax.bar(x=1.1, height=height, width=0.2, color=color, bottom=reg[0], edgecolor='w')

    sec_ax = ax.secondary_yaxis('right')
    sec_ax.set_yticks(np.mean(region, axis=1))
    sec_ax.set_yticklabels(np.around(scale, 2))
    sec_ax.tick_params(axis="y", direction="in")
    sec_ax.set_ylim([20, 3840])

# Instantiate brain atlas and one
brain_atlas = atlas.AllenAtlas(25)
one = ONE()

fig_path = Path('/Users/gaelle/Desktop/Alignment/')

# Find eid of interest
traj = one.alyx.rest('trajectories', 'list', provenance='Planned',
                     x=-2243, y=-2000,  # repeated site coordinate
                     project='ibl_neuropixel_brainwide_01',
                     django='probe_insertion__session__qc__lt,50')  # All except CRITICAL
probe_insertion = [t['probe_insertion'] for t in traj]

for p in probe_insertion:
    trajectory = one.alyx.rest('trajectories', 'list', provenance='Ephys aligned histology track',
                                probe_insertion=p)
    if len(trajectory) == 0:
        traj_p = one.alyx.rest('trajectories', 'list', provenance='Planned', probe_insertion=p)
        subject = traj_p[0]['session']['subject']
        print(f'No alignment for session : {p} - {subject}')
    else:
        # Plot
        eid = trajectory[0]['session']['id']
        probe_label = trajectory[0]['probe_name']
        subject = trajectory[0]['session']['subject']
        date = trajectory[0]['session']['start_time'][0:10]
        chn_coords = one.load(eid, dataset_types=['channels.localCoordinates'])[0]
        depths = chn_coords[:, 1]

        insertion = one.alyx.rest('insertions', 'list', session=eid, name=probe_label)
        xyz_picks = np.array(insertion[0]['json']['xyz_picks']) / 1e6

        alignments = trajectory[0]['json']

        # Code below taken from other code (compare_alignments)
        fig, ax = plt.subplots(1, len(alignments)+1, figsize=(15, 15))
        ephysalign = EphysAlignment(xyz_picks, depths)
        feature, track, _ = ephysalign.get_track_and_feature()
        channels_orig = ephysalign.get_channel_locations(feature, track)
        region, region_label = ephysalign.scale_histology_regions(feature, track)
        region_scaled, scale_factor = ephysalign.get_scale_factor(region)
        region_colour = ephysalign.region_colour

        norm = matplotlib.colors.Normalize(vmin=0.5, vmax=1.5, clip=True)
        mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.seismic)

        ax_i = fig.axes[0]
        plot_regions(region, region_label, region_colour, ax_i)
        plot_scaling(region_scaled, scale_factor, mapper, ax_i)
        ax_i.set_title('Original')

        for iK, key in enumerate(alignments):
            # Location of reference lines used for alignmnet
            feature = np.array(alignments[key][0])
            track = np.array(alignments[key][1])
            user = key[20:]
            # Instantiate EphysAlignment object
            ephysalign = EphysAlignment(xyz_picks, depths, track_prev=track, feature_prev=feature)

            channels = ephysalign.get_channel_locations(feature, track)
            avg_dist = np.mean(np.sqrt(np.sum((channels - channels_orig) ** 2, axis=1)), axis=0)
            region, region_label = ephysalign.scale_histology_regions(feature, track)
            region_scaled, scale_factor = ephysalign.get_scale_factor(region)

            ax_i = fig.axes[iK+1]
            plot_regions(region, region_label, region_colour, ax_i)
            plot_scaling(region_scaled, scale_factor, mapper, ax_i)
            ax_i.set_title(user + '\n Avg dist = ' + str(np.around(avg_dist * 1e6, 2)))

        fig.suptitle(subject + '_' + str(date) + '_' + probe_label, fontsize=16)
        plt.show()
        fig.savefig(fig_path.joinpath(subject + '_' + str(date) + '_' + probe_label + '.png'), dpi=600)
        plt.close(fig)

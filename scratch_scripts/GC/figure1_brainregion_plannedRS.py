"""
Fig 1 - Pannel C
Brain regions and coronal slice for Planned Traj of repeated site

# Note: this code outputs a warning: posx and posy should be finite values
# But it does generate the necessary figure

Authors: Gaelle C., Mayo F.
Aug. 2021
"""

from one.api import ONE
from iblatlas import atlas
import numpy as np
from ibllib.pipes import histology
from neuropixel import SITES_COORDINATES, TIP_SIZE_UM
from ibllib.pipes.ephys_alignment import EphysAlignment
import matplotlib.pyplot as plt


# Instantiate brain atlas and one
one = ONE()
brain_atlas = atlas.AllenAtlas(25)

# save path figure
path_save = '/Users/gaelle/Documents/Work/paper_reproducible_ephys/Fig_1/rs_traj_brainregion_plot.pdf'

# Get trajectory
trajectories = one.alyx.rest('trajectories', 'list', provenance='Planned',
                             x=-2243, y=-2000,  # repeated site coordinate
                             project='ibl_neuropixel_brainwide_01',
                             django='probe_insertion__session__qc__lt,50')
# Take first traj found, for example sake
traj = trajectories[0]

# Get channels from traj, inpired from:
# https://github.com/int-brain-lab/ibllib/blob/de3f451f43721e05677822a9e618a3bebe387e87/brainbox/io/one.py#L589-L593

ins = atlas.Insertion.from_dict(traj, brain_atlas=brain_atlas)
# Deepest coordinate first
depths = SITES_COORDINATES[:, 1]
xyz = np.c_[ins.tip, ins.entry].T
xyz_channels = histology.interpolate_along_track(xyz, (depths +
                                                       TIP_SIZE_UM) / 1e6)

# Get image, inpired from:
# https://int-brain-lab.github.io/iblenv/notebooks_external/docs_find_previous_alignments.html

xyz_picks = xyz_channels
ephysalign = EphysAlignment(xyz_picks, depths)

# Find brain region that each channel is located in
brain_regions = ephysalign.get_brain_locations(xyz_channels)

# For plotting -> extract the boundaries of the brain regions, as well as CCF label and colour
region, region_label, region_colour, _ = ephysalign.get_histology_regions(xyz_channels, depths)


# Create a figure and arrange using gridspec
widths = [1, 2.5]
heights = [1] * 1
gs_kw = dict(width_ratios=widths, height_ratios=heights)
fig, axis = plt.subplots(1, 2, constrained_layout=True,
                         gridspec_kw=gs_kw, figsize=(8, 9))

# Make plot that shows the brain regions that channels pass through
ax_regions = fig.axes[0]
for reg, col in zip(region, region_colour):
    height = np.abs(reg[1] - reg[0])
    bottom = reg[0]
    color = col / 255
    ax_regions.bar(x=0.5, height=height, width=1, color=color, bottom=reg[0], edgecolor='w')
ax_regions.set_yticks(region_label[:, 0].astype(int))
ax_regions.yaxis.set_tick_params(labelsize=8)
ax_regions.get_xaxis().set_visible(False)
ax_regions.set_yticklabels(region_label[:, 1])
ax_regions.spines['right'].set_visible(False)
ax_regions.spines['top'].set_visible(False)
ax_regions.spines['bottom'].set_visible(False)
ax_regions.hlines([0, 3840], *ax_regions.get_xlim(), linestyles='dashed', linewidth=3,
                  colors='k')
# ax_regions.plot(np.ones(channel_depths_track.shape), channel_depths_track, '*r')

# Make plot that shows coronal slice that trajectory passes through with location of channels
# shown in red
ax_slice = fig.axes[1]
brain_atlas.plot_tilted_slice(xyz_channels, axis=1, ax=ax_slice, volume='annotation')
ax_slice.plot(xyz_channels[:, 0] * 1e6, xyz_channels[:, 2] * 1e6, 'r*')
ax_slice.title.set_text('repeated site')


# Make sure the plot displays
plt.show()

# save
plt.savefig(path_save)

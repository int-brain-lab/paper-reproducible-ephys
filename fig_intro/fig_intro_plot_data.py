from reproducible_ephys_functions import save_figure_path, get_insertions, BRAIN_REGIONS, combine_regions
from iblatlas.atlas import AllenAtlas, Insertion
from ibllib.pipes import histology
from neuropixel import SITES_COORDINATES, TIP_SIZE_UM
import numpy as np
from brainbox.io.one import SpikeSortingLoader
from brainbox.plot import driftmap
from brainbox.ephys_plots import plot_brain_regions
from fig_intro.fig_intro_load_data import example_pid, suppl_pids
from fig_intro.fig_intro_functions import add_br_meshes
from one.api import ONE

import matplotlib.pyplot as plt
import itertools


def plot_main_figure(one, ba=None):
    ba = ba or AllenAtlas()

    plot_repeated_site_slice(one, ba=ba)
    plot_example_raster_with_regions(one, ba=ba)
    plot_3D_repeated_site_trajectories(one, ba=ba)


def plot_supp2_figure(one, ba=None):
    ba = ba or AllenAtlas()

    plot_3D_select_pids(one, ba=ba)
    plot_multiple_raster_with_regions(one, ba=ba)


def plot_repeated_site_slice(one, ba=None):
    ba = ba or AllenAtlas()
    ba.regions.rgb[0] = [255, 255, 255]
    insertions = get_insertions(level=0, one=one, freeze=None)
    traj = insertions[0]
    ins = Insertion.from_dict(traj, brain_atlas=ba)
    depths = SITES_COORDINATES[:, 1]
    xyz = np.c_[ins.tip, ins.entry].T
    xyz_channels = histology.interpolate_along_track(xyz, (depths + TIP_SIZE_UM) / 1e6)

    # Get the regions and colours for all the atlas ids
    region_info = ba.regions.get(ba.get_labels(xyz_channels))
    boundaries = np.where(np.diff(region_info.id) != 0)[0]
    boundaries = np.r_[0, boundaries, region_info.id.shape[0] - 1]
    regions = depths[np.c_[boundaries[0:-1], boundaries[1:]]]
    region_colours = region_info.rgb[boundaries[1:]]

    # Get the region labesl from the rep_site_acronyms
    regions_rep_acro = combine_regions(region_info.acronym)
    regions_rep_acro[regions_rep_acro == 'PPC'] = 'VISa'
    regions_rep_id = ba.regions.acronym2id(regions_rep_acro)
    boundaries = np.where(np.diff(regions_rep_id) != 0)[0]
    boundaries = np.r_[0, boundaries, region_info.id.shape[0] - 1]
    regions_rep = depths[np.c_[boundaries[0:-1], boundaries[1:]]]
    region_labels = np.c_[np.mean(regions_rep, axis=1).astype(int), regions_rep_acro[boundaries[1:]]]
    region_labels[region_labels[:, 1] == 'VISa', 1] = 'PPC'
    reg_idx = np.where(np.isin(region_labels[:, 1], BRAIN_REGIONS))[0]
    region_labels = region_labels[reg_idx, :]

    # Create a figure and arrange using gridspec
    widths = [1]
    heights = [1, 2.5]
    gs_kw = dict(width_ratios=widths, height_ratios=heights)
    fig, axs = plt.subplots(2, 1, constrained_layout=True, gridspec_kw=gs_kw, figsize=(8, 9))

    ba.plot_cslice(xyz_channels[0, 1], ax=axs[0], volume='annotation')
    axs[0].plot(xyz_channels[:, 0] * 1e6, xyz_channels[:, 2] * 1e6, 'k-')
    axs[0].set_axis_off()

    for reg, col in zip(regions, region_colours):
        height = np.abs(reg[1] - reg[0])
        color = col / 255
        axs[1].bar(x=0.5, height=height, width=1, color=color, bottom=reg[0], edgecolor='w')

    axs[1].set_yticks(region_labels[:, 0].astype(int))
    axs[1].set_yticklabels(region_labels[:, 1])
    axs[1].yaxis.set_tick_params(labelsize=12, pad=-250, length=0)
    axs[1].set_ylim(0, np.max(SITES_COORDINATES))
    axs[1].set_xlim(-5, 6)
    axs[1].get_xaxis().set_visible(False)
    axs[1].set_yticklabels(region_labels[:, 1])
    axs[1].spines['left'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['bottom'].set_visible(False)

    fig_path = save_figure_path(figure='fig_intro')
    plt.savefig(fig_path.joinpath('fig_intro_panelC.png'))
    plt.close()


def plot_3D_repeated_site_trajectories(one, ba=None):
    try:
        import mayavi.mlab as mlab
    except ModuleNotFoundError:
        print('Mayavi not installed, figure will not be generated')
        return

    ba = ba or AllenAtlas()
    fig = mlab.figure(bgcolor=(1, 1, 1))

    insertions = get_insertions(level=0, one=one, freeze=None)
    for ins in insertions:
        traj = one.alyx.rest('trajectories', 'list', provenance='Ephys aligned histology track',
                             probe_insertion=ins['probe_insertion'])[0]

        ins = Insertion.from_dict(traj, brain_atlas=ba)
        mlapdv = ba.xyz2ccf(ins.xyz)
        # display the trajectories
        color = (0., 0., 0.)
        mlab.plot3d(mlapdv[:, 1], mlapdv[:, 2], mlapdv[:, 0], line_width=3, tube_radius=20, color=color)
    add_br_meshes(fig, opacity=0.4)
    mlab.view(azimuth=180, elevation=0)
    mlab.view(azimuth=-160, elevation=111, reset_roll=False)
    fig_path = save_figure_path(figure='fig_intro')
    mlab.savefig(filename=str(fig_path.joinpath('fig_intro_panelD.png')))
    mlab.close()


def plot_example_raster_with_regions(one, pid=None, ba=None):

    ba = ba or AllenAtlas()

    widths = [1, 10]
    heights = [1]
    gs_kw = dict(width_ratios=widths, height_ratios=heights)
    fig, axs = plt.subplots(1, 2, constrained_layout=True, gridspec_kw=gs_kw, figsize=(15, 5))
    pid = pid or example_pid

    plot_raster_with_regions(pid, one, axs[1], axs[0], ba)

    axs[0].set_axis_off()
    axs[1].set_xticks([0, 1000, 2000, 3000])
    axs[1].set_yticks([0, 1000, 2000, 3000])
    axs[1].yaxis.set_tick_params(labelsize=14)
    axs[1].xaxis.set_tick_params(labelsize=14)
    axs[1].set_ylabel('Distance from probe tip (um)', size=16)
    axs[1].set_xlabel('Time (s)', size=16)

    fig_path = save_figure_path(figure='fig_intro')
    plt.savefig(fig_path.joinpath('fig_intro_panelE.png'))
    plt.close()


def plot_3D_select_pids(one, ba=None):

    try:
        import mayavi.mlab as mlab
    except ModuleNotFoundError:
        print('Mayavi not installed, figure will not be generated')
        return

    ba = ba or AllenAtlas()
    fig = mlab.figure(bgcolor=(1, 1, 1))

    pids = [ids for vals in suppl_pids.values() for ids in vals]

    colors = list(itertools.repeat((1, 0, 0), len(suppl_pids['GOOD']))) + \
             list(itertools.repeat((1.0, 0.49803922, 0.05490196), len(suppl_pids['MISS_TARGET']))) + \
             list(itertools.repeat((0, 0, 1), len(suppl_pids['LOW_YIELD'])))

    for ins, col in zip(pids, colors):
        traj = one.alyx.rest('trajectories', 'list', provenance='Ephys aligned histology track',
                             probe_insertion=ins)[0]

        ins = Insertion.from_dict(traj, brain_atlas=ba)
        mlapdv = ba.xyz2ccf(ins.xyz)
        # display the trajectories
        mlab.plot3d(mlapdv[:, 1], mlapdv[:, 2], mlapdv[:, 0], line_width=3, tube_radius=20, color=col)

    add_br_meshes(fig, brain_areas=['VISa', 'CA1', 'DG', 'LP', 'PO'], opacity=0.4)
    mlab.view(azimuth=180, elevation=0)
    mlab.view(azimuth=-160, elevation=111, reset_roll=False)
    mlab.view(azimuth=-46, elevation=107, reset_roll=False)
    fig_path = save_figure_path(figure='fig_intro')
    mlab.savefig(filename=str(fig_path.joinpath('fig_intro_supp2_3D.png')))
    mlab.close()


def plot_multiple_raster_with_regions(one, pids=None, ba=None):
    ba = ba or AllenAtlas()
    if pids is None:
        pids = [ids for vals in suppl_pids.values() for ids in vals]  # flat list of pids from dict

    ncols = 3
    nrows = int(np.ceil(len(pids) / ncols))

    fig = plt.figure(constrained_layout=False, figsize=(18, 10))
    subfigs = fig.subfigures(nrows, ncols)
    subfigs = subfigs.flat

    for i, pid in enumerate(pids):
        print(pid)
        widths = [10, 1]
        heights = [1]
        gs_kw = dict(width_ratios=widths, height_ratios=heights, wspace=0, hspace=0)
        axs = subfigs[i].subplots(1, 2, gridspec_kw=gs_kw)
        plot_raster_with_regions(pid, one, axs[0], axs[1], ba, mapping='Beryl', labels='right')
        axs[1].yaxis.set_tick_params(labelsize=12)
        if i == (nrows * ncols) - ncols:
            axs[0].set_xticks([0, 1000, 2000, 3000])
            axs[0].set_yticks([0, 1000, 2000, 3000])
            axs[0].yaxis.set_tick_params(labelsize=8)
            axs[0].xaxis.set_tick_params(labelsize=8)
            axs[0].set_ylabel('Distance from probe tip (um)', size=10)
            axs[0].set_xlabel('Time (s)', size=10)

        else:
            axs[0].set_xticklabels([])
            axs[0].set_yticklabels([])
            axs[0].set_ylabel('', size=12)
            axs[0].set_xlabel('', size=12)

    fig_path = save_figure_path(figure='fig_intro')
    plt.savefig(fig_path.joinpath('fig_intro_supp2_rasters.png'))
    plt.close()


def plot_raster_with_regions(pid, one, ax_raster, ax_regions, ba, mapping='Allen', restrict_labels=True, labels='left'):

    sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    spikes, clusters, channels = sl.load_spike_sorting(revision='2024-03-22')

    driftmap(spikes['times'], spikes['depths'], ax=ax_raster, plot_style='bincount')
    ax_raster.set_xlim([0, 60 * 60])  # display 1 hour
    ax_raster.set_ylim([0, np.max(channels['axial_um'])])

    atlas_ids = ba.regions.id2id(channels['atlas_id'], mapping=mapping)
    regions, region_labels, region_colours = plot_brain_regions(channel_ids=atlas_ids, channel_depths=channels['axial_um'],
                                                                brain_regions=ba.regions, display=False)
    if restrict_labels:
        # Remove any void or root region labels and that are less than 60 um
        reg_idx = np.where(~np.bitwise_or(np.isin(region_labels[:, 1], ['void', 'root']),
                                          (regions[:, 1] - regions[:, 0]) < 150))[0]
        region_labels = region_labels[reg_idx, :]

    for reg, col in zip(regions, region_colours):
        height = np.abs(reg[1] - reg[0])
        color = col / 255
        ax_regions.bar(x=0.5, height=height, width=1, color=color, bottom=reg[0], edgecolor='w')
    if labels == 'right':
        ax_regions.yaxis.tick_right()
    ax_regions.set_yticks(region_labels[:, 0].astype(int))
    ax_regions.yaxis.set_tick_params(labelsize=8)
    ax_regions.set_ylim(0, np.max(channels['axial_um']))
    ax_regions.get_xaxis().set_visible(False)
    ax_regions.set_yticklabels(region_labels[:, 1])
    ax_regions.spines['left'].set_visible(False)
    ax_regions.spines['right'].set_visible(False)
    ax_regions.spines['top'].set_visible(False)
    ax_regions.spines['bottom'].set_visible(False)


if __name__ == "__main__":
    one = ONE()
    plot_main_figure(one)

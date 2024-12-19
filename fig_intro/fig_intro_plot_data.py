from reproducible_ephys_functions import (save_figure_path, get_insertions, BRAIN_REGIONS,
                                          combine_regions, figure_style, get_label_pos, get_row_coord)
from iblatlas.atlas import AllenAtlas, Insertion
from ibllib.pipes import histology
from neuropixel import trace_header, TIP_SIZE_UM
from ibldsp.voltage import destripe
import numpy as np
import scipy.signal
from brainbox.io.one import SpikeSortingLoader
from brainbox.plot import driftmap
from brainbox.ephys_plots import plot_brain_regions
from fig_intro.fig_intro_load_data import example_pid, suppl_pids
from fig_intro.fig_intro_functions import add_br_meshes
from one.api import ONE
import figrid as fg
from fig_data_quality.fig_neuron_yield import plot_neuron_yield
from ibllib.plots.misc import Density

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.transforms import Bbox
import itertools

SITES_COORDINATES = trace_header(version=1)



def plot_main_figure(one, ba=None):
    ba = ba or AllenAtlas()
    width = 7
    height = 7
    figure_style()
    fig = plt.figure(figsize=(width, height), dpi=300)

    xspan_row1 = get_row_coord(width, [1])
    xspan_row2 = get_row_coord(width, [3, 1])
    xspan_row3 = get_row_coord(width, [2, 3, 3])
    yspans = get_row_coord(height, [1, 5, 2], pad=0.3)
    c_yspan = get_row_coord(height, [1, 2], pad=0, hspace=0.01, span=yspans[1])
    e_xspan = get_row_coord(width, [10, 1], pad=0, hspace=0, span=xspan_row3[1])
    f_xspan = get_row_coord(width, [1, 1, 1], pad=0, hspace=0.1, span=xspan_row3[2])

    ax = {'A': fg.place_axes_on_grid(fig, xspan=xspan_row1[0], yspan=yspans[0]),
          'B': fg.place_axes_on_grid(fig, xspan=xspan_row2[0], yspan=yspans[1]),
          'C_1': fg.place_axes_on_grid(fig, xspan=xspan_row2[1], yspan=c_yspan[0]),
          'C_2': fg.place_axes_on_grid(fig, xspan=xspan_row2[1], yspan=c_yspan[1]),
          'D': fg.place_axes_on_grid(fig, xspan=xspan_row3[0], yspan=yspans[2]),
          'E_1': fg.place_axes_on_grid(fig, xspan=e_xspan[0], yspan=yspans[2]),
          'E_2': fg.place_axes_on_grid(fig, xspan=e_xspan[1], yspan=yspans[2]),
          'F_1': fg.place_axes_on_grid(fig, xspan=f_xspan[0], yspan=yspans[2]),
          'F_2': fg.place_axes_on_grid(fig, xspan=f_xspan[1], yspan=yspans[2]),
          'F_3': fg.place_axes_on_grid(fig, xspan=f_xspan[2], yspan=yspans[2]),
          }

    ax['A'].set_axis_off()
    ax['B'].set_axis_off()
    ax['D'].set_axis_off()

    plot_repeated_site_slice(one, ba=ba, axs=[ax['C_1'], ax['C_2']], save=False)
    plot_example_raster_with_regions(one, ba=ba, axs=[ax['E_2'], ax['E_1']], save=False)
    plot_neuron_yield(ax=[ax['F_1'], ax['F_2'], ax['F_3']], save=False)


    labels = [{'label_text': 'a', 'xpos': get_label_pos(width,xspan_row1[0][0]), 'ypos': get_label_pos(height, yspans[0][0], pad=0.3),
               'fontsize': 10, 'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'b', 'xpos': get_label_pos(width, xspan_row2[0][0]), 'ypos': get_label_pos(height, yspans[1][0], pad=0.3), 'fontsize': 10,
               'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'c', 'xpos': get_label_pos(width, xspan_row2[1][0]), 'ypos': get_label_pos(height, yspans[1][0], pad=0.3), 'fontsize': 10,
               'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'd', 'xpos': get_label_pos(width, xspan_row3[0][0]),
               'ypos': get_label_pos(height, yspans[2][0], pad=0.3), 'fontsize': 10,
               'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'e', 'xpos': get_label_pos(width, xspan_row3[1][0], pad=0.5),
               'ypos': get_label_pos(height, yspans[2][0], pad=0.3), 'fontsize': 10,
               'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'f', 'xpos': get_label_pos(width, xspan_row3[2][0], pad=0.5),
               'ypos': get_label_pos(height, yspans[2][0], pad=0.3), 'fontsize': 10,
               'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              ]
    fg.add_labels(fig, labels)

    fig_path = save_figure_path(figure='fig_intro')
    adjust = 0.3
    fig.subplots_adjust(top=1-adjust/height, bottom=(adjust + 0.2)/height, left=(adjust)/width, right=1-adjust/width)
    fig.savefig(fig_path.joinpath('fig_intro.pdf'))
    fig.savefig(fig_path.joinpath('fig_intro.svg'))


def plot_supp2_figure(one, ba=None):
    ba = ba or AllenAtlas()
    figure_style()
    width = 7
    height = 6
    fig = plt.figure(figsize=(width, height), dpi=300)

    xspan = get_row_coord(width, [1, 1], hspace=0.6)
    yspans = get_row_coord(height, [1, 1, 1], hspace=0.8)
    xspan_inset1 = get_row_coord(width, [10, 1], pad=0, hspace=0, span=xspan[0])
    xspan_inset2 = get_row_coord(width, [10, 1], pad=0, hspace=0, span=xspan[1])

    ax = {
        'A_1': fg.place_axes_on_grid(fig, xspan=xspan_inset1[0], yspan=yspans[0]),
        'A_2': fg.place_axes_on_grid(fig, xspan=xspan_inset1[1], yspan=yspans[0]),
        'A_3': fg.place_axes_on_grid(fig, xspan=xspan_inset2[0], yspan=yspans[0]),
        'A_4': fg.place_axes_on_grid(fig, xspan=xspan_inset2[1], yspan=yspans[0]),
        'B_1': fg.place_axes_on_grid(fig, xspan=xspan_inset1[0], yspan=yspans[1]),
        'B_2': fg.place_axes_on_grid(fig, xspan=xspan_inset1[1], yspan=yspans[1]),
        'B_3': fg.place_axes_on_grid(fig, xspan=xspan_inset2[0], yspan=yspans[1]),
        'B_4': fg.place_axes_on_grid(fig, xspan=xspan_inset2[1], yspan=yspans[1]),
        'C_1': fg.place_axes_on_grid(fig, xspan=xspan_inset1[0], yspan=yspans[2]),
        'C_2': fg.place_axes_on_grid(fig, xspan=xspan_inset1[1], yspan=yspans[2]),
        'C_3': fg.place_axes_on_grid(fig, xspan=xspan_inset2[0], yspan=yspans[2]),
        'C_4': fg.place_axes_on_grid(fig, xspan=xspan_inset2[1], yspan=yspans[2]),
    }

    # Good pid
    pid = '63517fd4-ece1-49eb-9259-371dc30b1dd6'
    eid, _ = one.pid2eid(pid)
    trials = one.load_object(eid, 'trials')
    plot_example_raster_with_regions(one, pid=pid, ba=ba, axs=[ax['A_2'], ax['A_1']], save=False)
    ax['A_1'].axvline(trials['intervals'][-1][1], color='k')

    ax['A_2'].text(0, 1.2, 'PASS', c='k', rotation_mode='anchor', ha='left', transform=ax['A_2'].get_xaxis_transform())

    # Plot the raw data
    plot_raw_data(one, pid, ba=ba, axs=[ax['A_3'], ax['A_4']])

    # Epilepsy pid
    pid = '5246af08-0730-40f7-83de-29b5d62b9b6d'
    eid, _ = one.pid2eid(pid)
    trials = one.load_object(eid, 'trials')
    plot_example_raster_with_regions(one, pid=pid, ba=ba, axs=[ax['B_2'], ax['B_1']], save=False)
    ax['B_1'].axvline(trials['intervals'][-1][1], color='k')
    la = trials['intervals'][-1][1]

    for x in [770, 1120, 1470, 1800, 2100, 2560, 3175, 3800]:
        ax['B_1'].annotate('', xy=(x, 3700), xytext=(x, 3720),
                           arrowprops=dict(facecolor='black', headwidth=5, headlength=7))
    ax['B_1'].text(2000, 1.1, 'Epileptiform activity', c='k',
                   rotation_mode='anchor', ha='center', transform=ax['B_1'].get_xaxis_transform())

    ax['B_2'].text(0, 1.2, 'FAIL', c='k', rotation_mode='anchor', ha='left', transform=ax['B_2'].get_xaxis_transform())

    # Drift pid
    pid = '341ef9bb-25f9-4eeb-8f1d-bdd054b22ba8'
    eid, _ = one.pid2eid(pid)
    trials = one.load_object(eid, 'trials')
    plot_example_raster_with_regions(one, pid=pid, ba=ba, axs=[ax['B_4'], ax['B_3']], save=False)
    ax['B_3'].axvline(trials['intervals'][-1][1], color='k')
    ax['B_3'].text(4850, 1.1, 'Drift', c='k',
                   rotation_mode='anchor', ha='center', transform=ax['B_3'].get_xaxis_transform())
    for x in [4850]:
        ax['B_3'].annotate('', xy=(x, 3700), xytext=(x, 3720),
                           arrowprops=dict(facecolor='black', headwidth=5, headlength=7))
    # Artifact pid
    pid = '8413c5c6-b42b-4ec6-b751-881a54413628'
    eid, _ = one.pid2eid(pid)
    trials = one.load_object(eid, 'trials')
    plot_example_raster_with_regions(one, pid=pid, ba=ba, axs=[ax['C_2'], ax['C_1']], save=False)
    ax['C_1'].axvline(trials['intervals'][-1][1], color='k')
    ax['C_1'].text(1800, 1.1, 'Artifacts', c='k',
                   rotation_mode='anchor', ha='center', transform=ax['C_1'].get_xaxis_transform())
    for x in [1300, 2300]:
        ax['C_1'].annotate('', xy=(x, 3700), xytext=(x, 3720),
                           arrowprops=dict(facecolor='black', headwidth=5, headlength=7))

    # Noisy channels
    pid = 'ef3d059a-59d5-4870-b355-563a8d7cfd2d'
    plot_raw_data(one, pid, ba=ba, axs=[ax['C_3'], ax['C_4']], save=False)
    ax['C_3'].text(50, 1.1, 'Noisy channels', c='k',
                   rotation_mode='anchor', ha='center', transform=ax['C_3'].get_xaxis_transform())

    adjust = 0.3
    fig.subplots_adjust(top=1-adjust/height, bottom=(adjust + 0.2)/height, left=(adjust + 0.2)/width, right=1-adjust/width)
    fig_path = save_figure_path(figure='fig_intro')
    fig.savefig(fig_path.joinpath('fig_intro_supp2.pdf'))


def plot_repeated_site_slice(one, ba=None, axs=None, save=True):
    ba = ba or AllenAtlas()
    ba.regions.rgb[0] = [255, 255, 255]
    insertions = get_insertions(level=0, one=one, freeze='freeze_2024_03')
    traj = insertions[0]
    ins = Insertion.from_dict(traj, brain_atlas=ba)
    depths = SITES_COORDINATES['y']
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
    region_labels[region_labels[:, 1] == 'PPC', 1] = 'VISa/am'

    # Create a figure and arrange using gridspec
    if axs is None:
        widths = [1]
        heights = [1, 2.5]
        gs_kw = dict(width_ratios=widths, height_ratios=heights)
        fig, axs = plt.subplots(2, 1, constrained_layout=True, gridspec_kw=gs_kw, figsize=(8, 9))
    else:
        fig = plt.gcf()

    ba.plot_cslice(xyz_channels[0, 1], ax=axs[0], volume='annotation')
    axs[0].plot(xyz_channels[:, 0] * 1e6, xyz_channels[:, 2] * 1e6, 'k-')
    axs[0].set_axis_off()

    for reg, col in zip(regions, region_colours):
        height = np.abs(reg[1] - reg[0])
        color = col / 255
        axs[1].bar(x=0.5, height=height, width=1, color=color, bottom=reg[0], edgecolor='w')

    axs[1].set_yticks(region_labels[:, 0].astype(int))
    axs[1].set_yticklabels(region_labels[:, 1])
    axs[1].yaxis.set_tick_params(pad=-30, length=0)
    axs[1].set_ylim(0, np.max(depths))
    axs[1].set_xlim(-2, 3)
    axs[1].get_xaxis().set_visible(False)
    axs[1].spines['left'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['bottom'].set_visible(False)

    if save:
        fig_path = save_figure_path(figure='fig_intro')
        plt.savefig(fig_path.joinpath('fig_intro_panelC.png'))
        plt.close()


def plot_raw_data(one, pid=None, t0=1000, ba=None, axs=None, save=True):

    ssl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    spikes, clusters, channels = ssl.load_spike_sorting(dataset_types=["spikes.samples"], revision="2024-03-22",
                                                        enforce_version=False)

    sr_ap = ssl.raw_electrophysiology(band="ap", stream=True)
    s0 = int(sr_ap.fs * t0)
    dur = int(0.09 * sr_ap.fs)  # We take 0.09 second of data
    raw_ap = sr_ap[s0:s0 + dur, :-sr_ap.nsync].T
    raw_ap = scipy.signal.sosfiltfilt(scipy.signal.butter(3, 300, 'hp', fs=sr_ap.fs, output='sos'), raw_ap)
    destriped = destripe(raw_ap, sr_ap.fs)#, channel_labels=channels.labels)

    # Get the spikes that are within this raw data snippet
    spike_selection = slice(*np.searchsorted(spikes.samples, [s0, s0 + dur]))
    su = spikes.clusters[spike_selection]
    sc = clusters.channels[spikes.clusters][spike_selection]
    ss = (spikes.samples[spike_selection] - s0) / sr_ap.fs * 1e3

    clusters_good = np.where(clusters.metrics.label == 1)[0]
    clusters_bad = np.where(clusters.metrics.label != 1)[0]

    inds_good = np.isin(su, clusters_good)
    inds_bad = np.isin(su, clusters_bad)

    sc_good = sc[inds_good]
    sc_bad = sc[inds_bad]
    ss_good = ss[inds_good]
    ss_bad = ss[inds_bad]

    if axs is None:
        fig, axs = plt.subplots(1, 2, gridspec_kw={
            'width_ratios': [.95, .05]}, figsize=(16, 9), sharex='col')
    else:
        fig = plt.gcf()

    Density(destriped, fs=sr_ap.fs, taxis=1, gain=-100., ax=axs[0])
    # We display goog units in green, bad units in red
    axs[0].scatter(ss_good, sc_good, color="green", alpha=0.75, s=0.1)
    axs[0].scatter(ss_bad, sc_bad, color="red", alpha=0.5, s=0.1)

    # Plot brain regions in another column
    plot_brain_regions(channels["atlas_id"], channel_depths=channels["axial_um"], ax=axs[1], display=True)
    axs[1].set_axis_off()
    axs[0].set_ylabel('Channels')
    axs[0].set_xticks([])
    axs[0].set_xticklabels([])
    scalebar = AnchoredSizeBar(axs[0].transData,
                               10, '10 ms', 'lower left',
                               pad=0.1,
                               color='k',
                               frameon=False,
                               size_vertical=1,
                               bbox_to_anchor=Bbox.from_bounds(0, -0.2, 0, 0),
                               bbox_transform=axs[0].get_xaxis_transform()
                               )
    axs[0].add_artist(scalebar)

    return fig


def plot_3D_repeated_site_trajectories(one, ba=None):
    try:
        import mayavi.mlab as mlab
    except ModuleNotFoundError:
        print('Mayavi not installed, figure will not be generated')
        return

    ba = ba or AllenAtlas()
    fig = mlab.figure(bgcolor=(1, 1, 1))

    insertions = get_insertions(level=0, one=one, freeze='freeze_2024_03')
    for ins in insertions:
        traj = one.alyx.rest('trajectories', 'list', provenance='Ephys aligned histology track',
                             probe_insertion=ins['probe_insertion'])[0]
        if traj['x'] > 0:
            continue

        ins = Insertion.from_dict(traj, brain_atlas=ba)
        mlapdv = ba.xyz2ccf(ins.xyz)
        # display the trajectories
        color = (0., 0., 0.)
        mlab.plot3d(mlapdv[:, 1], mlapdv[:, 2], mlapdv[:, 0], line_width=3, tube_radius=20, color=color)

    # Add the planned location
    traj = one.alyx.rest('trajectories', 'list', provenance='Planned',
                         probe_insertion=insertions[0]['probe_insertion'])[0]
    ins = Insertion.from_dict(traj, brain_atlas=ba)
    mlapdv = ba.xyz2ccf(ins.xyz)
    # display the trajectories
    color = (1., 0., 0.)
    mlab.plot3d(mlapdv[:, 1], mlapdv[:, 2], mlapdv[:, 0], line_width=5, tube_radius=40, color=color)

    add_br_meshes(fig, opacity=0.4)
    mlab.view(azimuth=180, elevation=0)
    mlab.view(azimuth=-160, elevation=111, reset_roll=False)
    fig_path = save_figure_path(figure='fig_intro')
    mlab.savefig(filename=str(fig_path.joinpath('fig_intro_panelD.png')), size=(1024, 1024))
    mlab.close()


def plot_example_raster_with_regions(one, pid=None, ba=None, axs=None, save=True):

    ba = ba or AllenAtlas()

    if axs is None:
        widths = [1, 10]
        heights = [1]
        gs_kw = dict(width_ratios=widths, height_ratios=heights)
        fig, axs = plt.subplots(1, 2, constrained_layout=True, gridspec_kw=gs_kw, figsize=(15, 5))
    else:
        fig = plt.gcf()

    if pid is None:
        pid = example_pid
        example = True
    else:
        example = False

    plot_raster_with_regions(pid, one, axs[1], axs[0], ba)

    axs[0].set_axis_off()
    maxx = axs[1].get_xlim()[1]
    axs[1].set_xticks(np.arange(0, maxx, 1000))
    axs[1].set_yticks([0, 1000, 2000, 3000])

    axs[1].set_ylabel('Depth (\u03bcm)')
    axs[1].set_xlabel('Time (s)')
    if example:
        # Only show the active period, not the passive period
        axs[1].set_xlim(0, 2657)

    if save:
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

    figure_style()
    fig = plt.figure(figsize=(7, 4), dpi=300)

    gs = gridspec.GridSpec(nrows, ncols+2, figure=fig, width_ratios=[2, 1, 2, 2, 2], wspace=0.4)

    ax_3D = fig.add_subplot(gs[:, 0])
    ax_3D.set_axis_off()

    ax_good = fig.add_subplot(gs[0:2, 1])
    ax_good.annotate('Good', xy=(0.6, 0.5), xytext=(0.4, 0.5), xycoords='axes fraction',
                fontsize=8, ha='right', va='center', rotation=90, color='darkorange',
                arrowprops=dict(arrowstyle='-[, widthB=6.0, lengthB=1.5', lw=1.5, color='darkorange'))
    ax_good.set_axis_off()


    ax_miss = fig.add_subplot(gs[2:3, 1])
    ax_miss.annotate('Missed target', xy=(0.6, 0.5), xytext=(0.4, 0.5), xycoords='axes fraction',
                fontsize=8, ha='right', va='center', rotation=90, color='b',
                arrowprops=dict(arrowstyle='-[, widthB=3.0, lengthB=1.5', lw=1.5, color='b'))

    ax_miss.set_axis_off()

    ax_low = fig.add_subplot(gs[3:4, 1])
    ax_low.annotate('Low yield', xy=(0.6, 0.5), xytext=(0.4, 0.5), xycoords='axes fraction',
                fontsize=8, ha='right', va='center', rotation=90, color='r',
                arrowprops=dict(arrowstyle='-[, widthB=3.0, lengthB=1.5', lw=1.5, color='r'))
    ax_low.set_axis_off()

    for i, pid in enumerate(pids):

        subfig = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[int(i / ncols), np.mod(i, ncols) + 2],
                                                  width_ratios=[10, 1], wspace=0)

        axs = [fig.add_subplot(subfig[0, 0]), fig.add_subplot(subfig[0, 1])]
        plot_raster_with_regions(pid, one, axs[0], axs[1], ba, mapping='Beryl', labels='right')
        axs[1].yaxis.set_tick_params(length=0)
        if i == (nrows * ncols) - ncols:
            axs[0].set_xticks([0, 1000, 2000, 3000])
            axs[0].set_yticks([0, 1000, 2000, 3000])
            axs[0].set_ylabel('Depth (\u03bcm)')
            axs[0].set_xlabel('Time (s)')

        else:
            axs[0].set_xticklabels([])
            axs[0].set_yticklabels([])
            axs[0].set_ylabel('')
            axs[0].set_xlabel('')

    fig_path = save_figure_path(figure='fig_intro')
    fig.savefig(fig_path.joinpath('fig_intro_supp2_rasters.png'), bbox_inches='tight')
    fig.savefig(fig_path.joinpath('fig_intro_supp2_rasters.pdf'), bbox_inches='tight')
    plt.close()


def plot_raster_with_regions(pid, one, ax_raster, ax_regions, ba, mapping='Allen', restrict_labels=True, labels='left'):

    sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    spikes, clusters, channels = sl.load_spike_sorting(revision='2024-03-22', enforce_version=False)

    driftmap(spikes['times'], spikes['depths'], ax=ax_raster, plot_style='bincount')
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
    # plot_3D_repeated_site_trajectories(one)
    # plot_supp2_figure(one)


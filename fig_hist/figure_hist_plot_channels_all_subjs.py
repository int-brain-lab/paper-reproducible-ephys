#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 13:33:54 2021

This Python Script generates a figure containing all subject histology plots
coronal/sagittal through the repeated site histology data, and plots the
probe channels onto this histology data.

@author: sjwest
"""

import matplotlib.pyplot as plt
from iblatlas.atlas import Insertion
import numpy as np
import iblatlas.atlas as atlas
from fig_hist.fig_hist_load_data import load_dataframe
from fig_hist.fig_hist_functions import df_to_traj_dict
from reproducible_ephys_functions import save_figure_path, filter_recordings
from reproducible_ephys_functions import figure_style
from iblutil.numerical import ismember


def plot_channels(figcor=None, figsag=None, subjects=None, n_cols=None, remove_axes=False, show_scalebar=False, marker_size=None):

    import atlaselectrophysiology.load_histology as hist  # noqa
    # Load data
    probe_data = load_dataframe(df_name='traj')
    chn_data = load_dataframe(df_name='chns')

    # get new atlas for plotting
    brain_atlas = atlas.AllenAtlas(res_um=25)

    if subjects is not None:
        # If specific list of subjects is specified filter the dataframe
        probe_data = probe_data.loc[probe_data['subject'].isin(subjects)]
    else:
        # if none get all subjects from trajectory data
        subjects = probe_data['subject'].tolist()

    n_cols = n_cols or np.min([len(probe_data), 14]).astype(int)
    n_rows = np.ceil(len(probe_data) / n_cols).astype(int)

    # create figures
    figcor = figcor or plt.figure(figsize=(60, 20), dpi=72)
    figsag = figsag or plt.figure(figsize=(60, 20), dpi=72)

    mscor = marker_size or (72. / figcor.dpi) / 2
    mssag = marker_size or (72. / figcor.dpi) / 2

    # loop through all ids and generate tilted slice plots with recording sites plotted
    row_index = -1

    for i, (idx, row) in enumerate(probe_data.iterrows()):

        # keep track of row/col
        if np.mod(i, n_cols) == 0:
            row_index = row_index + 1
        col_index = np.mod(i, n_cols)

        axc = figcor.add_subplot(n_rows, n_cols, i + 1)
        axs = figsag.add_subplot(n_rows, n_cols, i + 1)

        traj = df_to_traj_dict(row, provenance='hist')
        ins = atlas.Insertion.from_dict(traj, brain_atlas)

        # traj and insertion of planned
        trajP = df_to_traj_dict(row, provenance='planned')
        insP = atlas.Insertion.from_dict(trajP, brain_atlas)

        # Download the histology data
        hist_paths = hist.download_histology_data(row['subject'], row['lab'])
        # create the brain atlases from the data
        ba_gr = atlas.AllenAtlas(hist_path=hist_paths[0])  # green histology channel autofl.
        ba_rd = atlas.AllenAtlas(hist_path=hist_paths[1])  # red histology channel cm-dii

        # in Z slices run from ANTERIOR to POSTERIOR (528-150,200)
        gr_thal_roi = ba_gr.image[328:378, 178:278, 100:200]  # isolate large slice over thalamus for max pixel value

        # Coronal slice
        axc, sec_axc = plot_slice(ba_gr, ba_rd, 1, ins, axc, roi=gr_thal_roi)
        # crop coronal figure to +/- 1000µm around the track
        xmn = np.min(ins.xyz[:, 0]) * 1e6 - 1000
        xmz = np.max(ins.xyz[:, 0]) * 1e6 + 1000
        axc.set_xlim(xmn, xmz)
        axc.axes.set_aspect('equal')  # ensure the resized xlim is not stretched!

        # Sagittal slice
        axs, sec_axs = plot_slice(ba_gr, ba_rd, 0, ins, axs, roi=gr_thal_roi)
        # crop sagittal figure to +/- 1000µm around the track
        xmn = np.min(ins.xyz[:, 1]) * 1e6 - 1000
        xmz = np.max(ins.xyz[:, 1]) * 1e6 + 1000
        axs.set_xlim(xmn, xmz)
        axs.axes.set_aspect('equal')  # ensure the resized xlim is not stretched!

        # Plot channels on each slice
        chns = chn_data[chn_data['pid'] == row['pid']]
        # retrieve the location in XYZ
        locX = chns['chan_loc_x'].values
        locY = chns['chan_loc_y'].values
        locZ = chns['chan_loc_z'].values

        # plot channels as circles at half the dpi
        axc.plot(locX * 1e6, locZ * 1e6, marker='o', ms=mscor, mew=0, color='w', linestyle="", lw=0)
        axs.plot(locY * 1e6, locZ * 1e6, marker='o', ms=mssag, mew=0, color='w', linestyle="", lw=0)

        # plot planned trajectory

        # remove all axis
        if remove_axes:
            axc.set_axis_off()
            sec_axc.set_axis_off()
            axs.set_axis_off()
            sec_axs.set_axis_off()
        else:
            # Only keep left y axis for first column
            if col_index != 0:
                axc.get_yaxis().set_visible(False)
                axs.get_yaxis().set_visible(False)
            # Only keep right y axis for last column
            if col_index != n_cols - 1:
                sec_axc.get_yaxis().set_visible(False)
                sec_axs.get_yaxis().set_visible(False)
            # Only keep x axis for last row
            if row_index != n_rows - 1:
                axc.get_xaxis().set_visible(False)
                axs.get_xaxis().set_visible(False)

        if show_scalebar:
            if i == 0:
                axc.plot([-500, -1000], [-7250, -7250], color='w', linewidth=1)

    return figcor, figsag


def plot_slice(ba_gr, ba_rd, axis, ins, plt_ax, roi=None, cmap=None, **kwargs):
    """
    Plot tilted slice of combined red and green images

    :param ba_gr: brain_atlas object with green histology input
    :param ba_rd: brain_atlas object with red histology input
    :param axis:  0: along ml = sagittal-slice 1: along ap = coronal-slice 2: along dv = horizontal-slice
    :param ins: insertion object of probe
    :param plt_ax: axis to plot on
    :param thal_roi: roi of green image to isolate intensity
    :param cmap: cmap to use
    :param kwargs:
    :return:
    """

    gr_percentile_min = kwargs.get('gr_percentile_min', 0.5)
    gr_percentile_max = kwargs.get('gr_percentile_max', 99.9)
    rd_percentile_min = kwargs.get('rd_percentile_min', 1)
    rd_percentile_max = kwargs.get('rd_percentile_max', 99.9)
    font_size = kwargs.get('font_size', 6)
    label_size = kwargs.get('label_size', 7)
    cmap = cmap or 'bone'

    # implementing tilted slice here to modify its cmap
    # get tilted slice of the green and red channel brain atlases
    gr_tslice, width, height, depth = ba_gr.tilted_slice(ins.xyz, axis, volume=ba_gr.image)
    rd_tslice, width, height, depth = ba_rd.tilted_slice(ins.xyz, axis, volume=ba_rd.image)

    gr_thal_roi = roi if roi is not None else gr_tslice

    width = width * 1e6
    height = height * 1e6
    depth = depth * 1e6

    # get the transfer function from y-axis to squeezed axis for second axe
    ab = np.linalg.solve(np.c_[height, height * 0 + 1], depth)

    # linearly scale the values in 2d numpy arrays to between 0-255 (8bit)
    # Using gr_tslice min and gr_thal_roi max to scale autofl.
    # using rd_tslice min and percentile (99.99 default) to scale CM-DiI
    gr_in = np.interp(gr_tslice, (np.percentile(gr_tslice, gr_percentile_min), np.percentile(gr_thal_roi, gr_percentile_max)),
                      (0, 255))
    rd_in = np.interp(rd_tslice, (np.percentile(rd_tslice, rd_percentile_min), np.percentile(rd_tslice, rd_percentile_max)),
                      (0, 255))

    # join together red, green, blue numpy arrays to form a RGB image ALONG A NEW DIMENSION
    # NOTE need a blue component, have added a set of zeros as blue channel should be BLANK
    # NOTE2: converted to unit8 bit, as pyplot imshow() method only reads this format
    Z = np.stack([rd_in.astype(dtype=np.uint8), gr_in.astype(dtype=np.uint8),
                  np.zeros(np.shape(gr_tslice)).astype(dtype=np.uint8)])
    # transpose the columns to the FIRST one is LAST  i.e the NEW DIMENSION [3] is the LAST DIMENSION
    Zt = np.transpose(Z, axes=[1, 2, 0])

    # can now add the RGB array to imshow()
    plt_ax.imshow(Zt, interpolation='none', aspect='auto', extent=np.r_[width, height], cmap=cmap, vmin=np.min(gr_in),
                  vmax=np.max(gr_in))
    sec_ax = plt_ax.secondary_yaxis('right', functions=(lambda x: x * ab[0] + ab[1], lambda y: (y - ab[1]) / ab[0]))

    if axis == 0:  # sagittal
        axis_labels = np.array(['ap (um)', 'dv (um)', 'ml (um)'])
    elif axis == 1:  # coronal
        axis_labels = np.array(['ml (um)', 'dv (um)', 'ap (um)'])
    elif axis == 2:
        axis_labels = np.array(['ml (um)', 'ap (um)', 'dv (um)'])

    plt_ax.set_xlabel(axis_labels[0], fontsize=font_size)
    plt_ax.set_ylabel(axis_labels[1], fontsize=font_size)
    plt_ax.set_ylabel(axis_labels[2], fontsize=font_size)

    plt_ax.tick_params(axis='x', labelrotation=90)
    plt_ax.tick_params(axis='x', labelsize=label_size)
    plt_ax.tick_params(axis='y', labelsize=label_size)
    sec_ax.tick_params(axis='y', labelsize=label_size)

    return plt_ax, sec_ax


def plot_all_channels(subjects=None, remove_exclusions=True):
    """
    Plot all subjects CORONAL & SAGITTAL histology and channels for repeated site
    Plots all coronal and all sagittal data in one large figure.
    :return:
    """

    if subjects is None:
        # grab subjects from trajectory data 
        probe_data = load_dataframe('traj')
        if remove_exclusions == True: # after exclusions
            df = filter_recordings()
            df = filter_recordings(min_neuron_region=0)
            # Find the pids are that are passing the inclusion criteria
            pids = df[df['include'] == 1]['pid'].unique()
            isin, _ = ismember(probe_data.pid.values, pids)
            probe_data['include'] = isin
            probe_data['passed'] = np.full(len(probe_data), 'PASS')
            probe_data.loc[~probe_data['include'], 'passed'] = 'FAIL'
            subjects = probe_data['subject'][probe_data['include'] == 1]
        else: # wihtout exclusions
            subjects = probe_data['subject']
    
    # create figures
    figcor = plt.figure(figsize=(20, 12), dpi=72)
    figsag = plt.figure(figsize=(20, 12), dpi=72)

    figcor, figsag = plot_channels(figcor, figsag, subjects=subjects, n_cols = 15,
                                   remove_axes=True, show_scalebar=True)

    # adjust spacing
    #wspace = 0.3
    #hspace = 0.1

    #figcor.subplots_adjust(wspace, hspace)
    #figsag.subplots_adjust(wspace, hspace)

    figcor.tight_layout()
    figsag.tight_layout()

    # save to output
    fig_path = save_figure_path(figure='fig_hist')
    figcor.savefig(fig_path.joinpath('all_channels_subj_hist_coronal.svg'), bbox_inches="tight")
    figsag.savefig(fig_path.joinpath('all_channels_subj_hist_sagittal.svg'), bbox_inches="tight")



def plot_channels_n2():
    """
    Plot two subjects (one close and one far form planned traj), CORONAL & SAGITTAL histology and channels for repeated site
    Plots all coronal and all sagittal data in one figure.
    :return:
    """
    subjects = ['KS045', 'CSHL054']

    figcor = plt.figure()
    figsag = plt.figure()

    # reset the sizes
    figcor.set_size_inches(3, 2.15)
    figsag.set_size_inches(3, 2.15)

    figcor, figsag = plot_channels(figcor, figsag, subjects=subjects, remove_axes=True, show_scalebar=True, marker_size=0.3)

    # adjust spacing
    wspace = 0.3
    hspace = 0.1
    figcor.subplots_adjust(wspace, hspace)
    figsag.subplots_adjust(wspace, hspace)

    figcor.tight_layout()
    figsag.tight_layout()

    # save to output
    fig_path = save_figure_path(figure='fig_hist')
    figcor.savefig(fig_path.joinpath('B_channels_subj2_hist_coronal.svg'), bbox_inches="tight")
    figsag.savefig(fig_path.joinpath('B_channels_subj2_hist_sagittal.svg'), bbox_inches="tight")


def plot_channels_n1():
    """
    Plot one subject (one closest to planned traj: KS045), CORONAL & SAGITTAL histology and channels for repeated site
    Plots all coronal and all sagittal data in one figure.
    :return:
    """

    # use repo-ephys figure style
    figure_style()

    subjects = ['KS045']

    figcor = plt.figure()
    figsag = plt.figure()

    # reset the sizes
    figcor.set_size_inches(1.5, 2.15)
    figsag.set_size_inches(1.5, 2.15)

    figcor, figsag = plot_channels(figcor, figsag, subjects=subjects, remove_axes=True, show_scalebar=True, marker_size=0.3)

    # flip sagittal axes in X to fit panel C sagittal layout
    axsag = figsag.get_axes()[0]
    axsag.invert_xaxis()

    # adjust spacing
    #wspace = 0.3
    #hspace = 0.1
    #figcor.subplots_adjust(wspace, hspace)
    #figsag.subplots_adjust(wspace, hspace)
    axcor = figcor.axes[0]
    axcor.text(-3000, -7500, 'Coronal', style='italic', color='w')

    #axsag = figsag.axes[0]
    axsag.text(-1750, -7500, 'Sagittal', style='italic', color='w')

    figcor.tight_layout()
    figsag.tight_layout()

    # save to output
    fig_path = save_figure_path(figure='fig_hist')
    figcor.savefig(fig_path.joinpath('B_channels_subj1_hist_coronal.svg'), bbox_inches="tight")
    figsag.savefig(fig_path.joinpath('B_channels_subj1_hist_sagittal.svg'), bbox_inches="tight")



def plot_channels_n3():
    """
    Plot three subjects, CORONAL & SAGITTAL histology and channels for repeated site
    Plots all coronal and all sagittal data in one large figure.
    :return:
    """
    subjects = ['CSH_ZAD_026', 'KS045', 'CSHL054']

    figcor = plt.figure()
    figsag = plt.figure()

    # reset the sizes
    figcor.set_size_inches(3, 2.15)
    figsag.set_size_inches(3, 2.15)

    figcor, figsag = plot_channels(figcor, figsag, subjects=subjects, remove_axes=True, show_scalebar=True, marker_size=0.3)

    # adjust spacing
    wspace = 0.3
    hspace = 0.1
    figcor.subplots_adjust(wspace, hspace)
    figsag.subplots_adjust(wspace, hspace)

    figcor.tight_layout()
    figsag.tight_layout()

    # save to output
    fig_path = save_figure_path(figure='fig_hist')
    figcor.savefig(fig_path.joinpath('B_channels_subj3_hist_coronal.svg'), bbox_inches="tight")
    figsag.savefig(fig_path.joinpath('B_channels_subj3_hist_sagittal.svg'), bbox_inches="tight")


if __name__ == "__main__":
    plot_channels_n3()
    plot_all_channels()

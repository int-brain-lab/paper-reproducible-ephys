#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 13:33:54 2021

This Python Script generates a figure containing all histology trajectories
projected onto coronal & sagittal plots of CCF along the PLANNED repeated site
trajectory.

@author: sjwest
"""

from pathlib import Path
import os
from figure_histology import figure_hist_data as fhd
import matplotlib.pyplot as plt
from one.api import ONE
import numpy as np
import ibllib.atlas as atlas
from reproducible_ephys_functions import figure_style, labs, save_figure_path
from figure_histology.figure2_load_data import load_dataframe
from figure_histology.figure2_functions import df_to_traj_dict


def plot_trajs(plan_colour='w', lab_colour=True):
    '''Plot CCF in coronal & sagittal tilted slices along planned rep site traj
    and add histology trajs projections onto this plot.
    '''
    
    # load in data
    probe_data = load_dataframe(df_name='traj')

    ins_plan = atlas.Insertion.from_dict(df_to_traj_dict(probe_data.iloc[0], provenance='planned'))

    # get new atlas for plotting
    brain_atlas = atlas.AllenAtlas(res_um=25)

    # use repo-ephys figure style
    figure_style()
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig1.set_size_inches(1, 2.15)
    fig2.set_size_inches(1, 2.15)

    # empty numpy arrays for storing the entry point of probe into brain
    # and "exit point" i.e the probe tip!
    all_ins_entry = np.empty((0, 3))
    all_ins_exit = np.empty((0, 3))
    
    # generate initial plot of brain atlas in each plane using ins_plan to take the correct tilted slice for PLANNED TRAJECTORY
    cax = brain_atlas.plot_tilted_slice(ins_plan.xyz, axis=1, ax=ax1)
    sax = brain_atlas.plot_tilted_slice(ins_plan.xyz, axis=0, ax=ax2)
    
    # get institution map and colours
    lab_number_map, institution_map, institution_colors = labs()
    
    # Compute trajectory for each repeated site recording and plot on slice figures
    for idx, row in probe_data.iterrows():
        
        subj = row['subject']
        lab = row['lab']
        
        print(subj)
        print(lab)
        print(institution_map[lab])

        traj = df_to_traj_dict(row, provenance='hist')
        ins = atlas.Insertion.from_dict(traj)
    
        all_ins_entry = np.vstack([all_ins_entry, ins.xyz[0, :]])
        all_ins_exit = np.vstack([all_ins_exit, ins.xyz[1, :]])

        # Plot the trajectory for each repeated site recording
        # colour by institution_colors[institution_map[phi_lab]]
        if lab_colour:
            color = institution_colors[institution_map[lab]]
            cax.plot(ins.xyz[:, 0] * 1e6, ins.xyz[:, 2] * 1e6, color=color, linewidth = 0.8)
            sax.plot(ins.xyz[:, 1] * 1e6, ins.xyz[:, 2] * 1e6, color=color,linewidth = 0.8)
        else:
            # OR plot all trajectories the same colour - deepskyblue
            cax.plot(ins.xyz[:, 0] * 1e6, ins.xyz[:, 2] * 1e6, color='deepskyblue', linewidth=0.5, alpha=0.5)
            sax.plot(ins.xyz[:, 1] * 1e6, ins.xyz[:, 2] * 1e6, color='deepskyblue', linewidth=0.5, alpha=0.5)

    # TODO check if we can remove
    # Compute the mean trajectory across all repeated site recordings
    # entry_mean = np.mean(all_ins_entry, axis=0)
    # exit_mean = np.mean(all_ins_exit, axis=0)
    # ins_mean = np.r_[[entry_mean], [exit_mean]]
    # Only consider deviation in ML and AP directions for this analysis
    # entry_std = np.std(all_ins_entry, axis=0)
    # entry_std[2]=0
    # exit_std = np.std(all_ins_exit, axis=0)
    # exit_std[2]=0
    # ins_upper = np.r_[[entry_mean+entry_std], [exit_mean+exit_std]]
    # ins_lower = np.r_[[entry_mean-entry_std], [exit_mean-exit_std]]
    
    # Plot the average track across all repeated site recordings, in RED
    # cax.plot(ins_mean[:, 0] * 1e6, ins_mean[:, 2] * 1e6, 'orangered', linewidth=2)
    # sax.plot(ins_mean[:, 1] * 1e6, ins_mean[:, 2] * 1e6, 'orangered', linewidth=2)

    # add planned insertion ON TOP of the actual insertions, in WHITE
    cax.plot(ins_plan.xyz[:, 0] * 1e6, ins_plan.xyz[:, 2] * 1e6, plan_colour, linewidth=2)
    sax.plot(ins_plan.xyz[:, 1] * 1e6, ins_plan.xyz[:, 2] * 1e6, plan_colour, linewidth=2)
    
    ax1.set_ylim((-6000,500))
    ax1.set_xlim((-3000,0))
    
    ax2.set_ylim((-6000,500))
    ax2.set_xlim((-1000,-4000))
    
    ax1.tick_params(axis='x', labelrotation = 90)
    ax2.tick_params(axis='x', labelrotation = 90)
    
    # hide the axes
    ax1.set_axis_off()
    ax2.set_axis_off()
    ax1.get_children()[len(ax1.get_children())-2].set_axis_off()
    ax2.get_children()[len(ax1.get_children())-2].set_axis_off()

    ax1.plot([-1250, -250], [-5750, -5750], color='w', linewidth=2)

    fig1.tight_layout()
    fig2.tight_layout()
    
    fig_path = save_figure_path(figure='figure2')
    fig1.savefig(fig_path.joinpath('C_probe_trajs_ccf_coronal.svg'), bbox_inches="tight")
    fig2.savefig(fig_path.joinpath('C_probe_trajs_ccf_sagittal.svg'), bbox_inches="tight")
    

if __name__ == "__main__":
    plot_trajs()



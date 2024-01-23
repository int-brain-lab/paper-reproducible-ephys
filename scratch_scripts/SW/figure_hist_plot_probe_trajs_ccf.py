#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 13:33:54 2021

This Python Script generates a figure containing all histology trajectories
projected onto coronal & sagittal plots of CCF along the PLANNED repeated site
trajectory.

@author: sjwest
"""

def print_path():
    import os
    path = os.path.dirname(os.path.realpath(__file__))
    print(path)


def plot_trajs(probe_data, output='figure_histology', 
               plan_colour='w', lab_colour=True):
    '''Plot CCF in coronal & sagittal tilted slices along planned rep site traj
    and add histology trajs projections onto this plot.
    '''
    from pathlib import Path
    import os
    import figure_hist_data as fhd
    import matplotlib.pyplot as plt
    from one.api import ONE
    import numpy as np
    import iblatlas.atlas as atlas
    import reproducible_ephys_functions as ref
    
    # use repo-ephys figure style
    ref.figure_style()
    
    # get probe data if necessary
    if 'probe_data' not in vars() or 'probe_data' not in globals():
        # get probe_data histology query
        probe_data = fhd.get_probe_data()
    
    traj_data = probe_data
    
    # output DIR - generate output DIR for storing plots:
    OUTPUT = Path(output)
    if os.path.exists(OUTPUT) is False:
        os.mkdir(OUTPUT)
    
    # connect to ibl server
    one = ONE()
    
    # get all trajectories at the REPEATED SITE
     # for ibl brainwide map project
     # repeated site location: x -2243 y -2000
    traj = one.alyx.rest('trajectories', 'list', provenance='Planned',
                         x=-2243, y=-2000,  project='ibl_neuropixel_brainwide_01')
    
    # get eids, probe names and subject names from traj
    eids = [sess['session']['id'] for sess in traj]
    probes = [sess['probe_name'] for sess in traj]
    
    # Get the trajectory for the planned repeated site recording
    phi_eid = eids[0]
    phi_probe = probes[0]
    phi_traj = one.alyx.rest('trajectories', 'list', session=phi_eid,
                             provenance='Planned', probe=phi_probe)[0]
    # planned as insertion: includes XYZ and phi,theta,depth data
    ins_plan = atlas.Insertion.from_dict(phi_traj)
    
    # get new atlas for plotting
    brain_atlas = atlas.AllenAtlas(res_um=25)
    
    
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    
    fig1.set_size_inches(1, 2.15)
    fig2.set_size_inches(1, 2.15)
    
    # set font sizes
    #SMALL_SIZE = 7
    #MEDIUM_SIZE = 8
    #BIGGER_SIZE = 10
    
    #plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    #plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    #plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    #plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    #plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    #plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    #plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    
    # Compute mean & deviation in coronal plane
    #alpha_mean = np.mean(np.abs(traj_data['planned_theta'] -
    #                            np.abs(traj_data['hist_coronal_angle'])))
    #alpha_std = np.std(np.abs(traj_data['planned_theta'] -
    #                          np.abs(traj_data['hist_coronal_angle'])))
    
    # Compute mean & deviation in sagittal plane
    #beta_mean = np.mean(np.abs(traj_data['hist_saggital_angle']))
    #beta_std = np.std(np.abs(traj_data['hist_saggital_angle']))
    
    # empty numpy arrays for storing the entry point of probe into brain
    # and "exit point" i.e the probe tip!
    all_ins_entry = np.empty((0, 3))
    all_ins_exit = np.empty((0, 3))
    
    # generate initial plot of brain atlas in each plane
     # using ins_plan to take the correct tilted slice for PLANNED TRAJECTORY
    cax = brain_atlas.plot_tilted_slice(ins_plan.xyz, axis=1, ax=ax1)
    sax = brain_atlas.plot_tilted_slice(ins_plan.xyz, axis=0, ax=ax2)
    
    # get institution map and colours
    lab_number_map, institution_map, institution_colors = ref.labs()
    
    # Compute trajectory for each repeated site recording and plot on slice figures
    for idx, row in traj_data.iterrows():
        
        phi_eid = row['eid']
        phi_probe = row['probe']
        phi_subj = row['subject']
        phi_lab = row['lab']
        
        print(phi_subj)
        print(phi_lab)
        print(institution_map[phi_lab])
        
        phi_traj = one.alyx.rest('trajectories', 'list', session=phi_eid,
                                 provenance='Histology track', probe=phi_probe)[0]
        ins = atlas.Insertion.from_dict(phi_traj)
    
        all_ins_entry = np.vstack([all_ins_entry, ins.xyz[0, :]])
        all_ins_exit = np.vstack([all_ins_exit, ins.xyz[1, :]])
        # channels = bbone.load_channel_locations(phi_eid, one=one,
        # probe=phi_probe)
    
        # Plot the trajectory for each repeated site recording
        # colour by institution_colors[institution_map[phi_lab]]
        if lab_colour:
            cax.plot(ins.xyz[:, 0] * 1e6, ins.xyz[:, 2] * 1e6, 
                     color= institution_colors[institution_map[phi_lab]],
                     linewidth = 0.8)
            sax.plot(ins.xyz[:, 1] * 1e6, ins.xyz[:, 2] * 1e6, 
                     color= institution_colors[institution_map[phi_lab]],
                     linewidth = 0.8)
        else:
            # OR plot all trajectories the same colour - deepskyblue
            cax.plot(ins.xyz[:, 0] * 1e6, ins.xyz[:, 2] * 1e6, 
                     color= 'deepskyblue',
                     linewidth = 0.5, alpha = 0.5)
            sax.plot(ins.xyz[:, 1] * 1e6, ins.xyz[:, 2] * 1e6, 
                     color= 'deepskyblue',
                     linewidth = 0.5, alpha=0.5)
    
    
    # Compute the mean trajectory across all repeated site recordings
    entry_mean = np.mean(all_ins_entry, axis=0)
    exit_mean = np.mean(all_ins_exit, axis=0)
    ins_mean = np.r_[[entry_mean], [exit_mean]]
    # Only consider deviation in ML and AP directions for this analysis
    # entry_std = np.std(all_ins_entry, axis=0)
    # entry_std[2]=0
    # exit_std = np.std(all_ins_exit, axis=0)
    # exit_std[2]=0
    # ins_upper = np.r_[[entry_mean+entry_std], [exit_mean+exit_std]]
    # ins_lower = np.r_[[entry_mean-entry_std], [exit_mean-exit_std]]
    
    # Plot the average track across all repeated site recordings, in RED
    #cax.plot(ins_mean[:, 0] * 1e6, ins_mean[:, 2] * 1e6, 'orangered', linewidth=2)
    #sax.plot(ins_mean[:, 1] * 1e6, ins_mean[:, 2] * 1e6, 'orangered', linewidth=2)

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
    #ax1.secondary_yaxis('right').axes.set_axis_off()
    #ax2.secondary_yaxis('right').axes.set_axis_off()
    #ax1.twinx().set_axis_off()
    #ax2.twinx().set_axis_off()
    
    ax1.plot(  [-1250, -250], 
               [ -5750, -5750 ], 
               color= 'w', linewidth = 2)
    #ax2.plot(  [-2750, -3750], 
    #           [ -5750, -5750 ], 
    #           color= 'w', linewidth = 2)
    
    
    #start, end = ax1.get_xlim()
    #ax1.xaxis.set_ticks(np.arange(start, end, 1000)) # ticks every 1000um
    #start, end = ax2.get_xlim()
    #ax2.xaxis.set_ticks(np.arange(start, end, -1000)) # ticks every 1000um
    
    ax1.text(-2900, -5700, 'Coronal', style='italic', color = 'w')
    ax2.text(-1100, -5700, 'Sagittal', style='italic', color = 'w')
    
    fig1.tight_layout()
    fig2.tight_layout()
    
    #plt.axis('off')
    # save to output
    fig1.savefig( str(Path(OUTPUT, 'C_probe_trajs_ccf_coronal.svg')), bbox_inches="tight" )
    fig2.savefig( str(Path(OUTPUT, 'C_probe_trajs_ccf_sagittal.svg')), bbox_inches="tight" )
    


def old_plot_trajs_cor_sag_hor(output='figure_histology'):
    '''Plot CCF in coronal & sagittal tilted slices along planned rep site traj
    and add histology trajs projections onto this plot.
    '''
    from pathlib import Path
    import os
    from probe_geometry_analysis import probe_geometry_data as probe_geom_data
    import matplotlib.pyplot as plt
    from one.api import ONE
    from iblatlas.atlas import Insertion
    import numpy as np
    import atlaselectrophysiology.load_histology as hist
    import iblatlas.atlas as atlas
    
    # output DIR
    OUTPUT = Path(output)
    
    # generate output DIR for storing plots:
    if os.path.exists(OUTPUT) is False:
        os.mkdir(OUTPUT)
    
    # connect to ibl server
    one = ONE()
    
    # get all trajectories at the REPEATED SITE
     # for ibl brainwide map project
     # repeated site location: x -2243 y -2000
    traj = one.alyx.rest('trajectories', 'list', provenance='Planned',
                         x=-2243, y=-2000,  project='ibl_neuropixel_brainwide_01')
    
    # get eids, probe names and subject names from traj
    eids = [sess['session']['id'] for sess in traj]
    probes = [sess['probe_name'] for sess in traj]
    
    # Get the trajectory for the planned repeated site recording
    phi_eid = eids[0]
    phi_probe = probes[0]
    phi_traj = one.alyx.rest('trajectories', 'list', session=phi_eid,
                             provenance='Planned', probe=phi_probe)[0]
    # planned as insertion: includes XYZ and phi,theta,depth data
    ins_plan = atlas.Insertion.from_dict(phi_traj)
    
    # get new atlas for plotting
    brain_atlas = atlas.AllenAtlas(res_um=25)
    
    
    # load trajectory data for repeated site from local cache
    traj_data = probe_geom_data.load_trajectory_data('-2243_-2000')
    
    # get subject IDs as list
    #ids = list(dict.fromkeys(traj_data['subject']))
    
    # manual exclusions
    index = traj_data[ traj_data['subject'] == 'CSH_ZAD_001' ].index
    traj_data.drop(index , inplace=True)
    index = traj_data[ traj_data['subject'] == 'CSHL051' ].index
    traj_data.drop(index , inplace=True)
    index = traj_data[ traj_data['subject'] == 'NYU-47' ].index
    traj_data.drop(index , inplace=True)
    index = traj_data[ traj_data['subject'] == 'UCLA011' ].index
    traj_data.drop(index , inplace=True)
    index = traj_data[ traj_data['subject'] == 'KS051' ].index
    traj_data.drop(index , inplace=True)
    #ids.remove('CSH_ZAD_001') # this has very poor anatomy - dissection damage
    #ids.remove('CSHL051') # this has very poor anatomy - dissection damage
    #ids.remove('NYU-47') # it has no histology ???
    #ids.remove('UCLA011') # currently repeated site WAYYY OUT!
    #ids.remove('KS051') # this sample has not been imaged..
    
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()
    
    # set font sizes
    SMALL_SIZE = 7
    MEDIUM_SIZE = 8
    BIGGER_SIZE = 10
    
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    
    # Compute mean & deviation in coronal plane
    alpha_mean = np.mean(np.abs(traj_data['planned_theta'] -
                                np.abs(traj_data['hist_coronal_angle'])))
    alpha_std = np.std(np.abs(traj_data['planned_theta'] -
                              np.abs(traj_data['hist_coronal_angle'])))
    
    # Compute mean & deviation in sagittal plane
    beta_mean = np.mean(np.abs(traj_data['hist_saggital_angle']))
    beta_std = np.std(np.abs(traj_data['hist_saggital_angle']))
    
    # empty numpy arrays for storing the entry point of probe into brain
    # and "exit point" i.e the probe tip!
    all_ins_entry = np.empty((0, 3))
    all_ins_exit = np.empty((0, 3))
    
    # generate initial plot of brain atlas in each plane
     # using ins_plan to take the correct tilted slice for PLANNED TRAJECTORY
    cax = brain_atlas.plot_tilted_slice(ins_plan.xyz, axis=1, ax=ax1)
    sax = brain_atlas.plot_tilted_slice(ins_plan.xyz, axis=0, ax=ax2)
    #hax = brain_atlas.plot_tilted_slice(ins_plan.xyz, axis=2, ax=ax3)
     # hax and hax2 are the SURFACE nad PROBE TIP horizontal sections
       # for PLANNED TRAJECTORY!
    hax = brain_atlas.plot_hslice(ins_plan.xyz[0, 2]-500/1e6, ax=ax3)
    hax2 = brain_atlas.plot_hslice(ins_plan.xyz[1, 2]-500/1e6, ax=ax4)
    
    # Compute trajectory for each repeated site recording and plot on slice figures
    for idx, row in traj_data.iterrows():
        phi_eid = row['eid']
        phi_probe = row['probe']
        phi_subj = row['subject']
        print(phi_subj)
        phi_traj = one.alyx.rest('trajectories', 'list', session=phi_eid,
                                 provenance='Histology track', probe=phi_probe)[0]
        ins = atlas.Insertion.from_dict(phi_traj)
    
        all_ins_entry = np.vstack([all_ins_entry, ins.xyz[0, :]])
        all_ins_exit = np.vstack([all_ins_exit, ins.xyz[1, :]])
        # channels = bbone.load_channel_locations(phi_eid, one=one,
        # probe=phi_probe)
    
        # Plot the trajectory for each repeated site recording
        cax.plot(ins.xyz[:, 0] * 1e6, ins.xyz[:, 2] * 1e6, 'deepskyblue', alpha=0.3)
        sax.plot(ins.xyz[:, 1] * 1e6, ins.xyz[:, 2] * 1e6, 'deepskyblue', alpha=0.3)
        hax.plot(ins.xyz[0, 1] * 1e6, ins.xyz[0, 0] * 1e6, color='deepskyblue', marker="o",
                 markersize=1, alpha=0.6)
        hax2.plot(ins.xyz[1, 1] * 1e6, ins.xyz[1, 0] * 1e6, color='deepskyblue', marker="o",
                  markersize=1, alpha=0.6)
    
    
    # Compute the mean trajectory across all repeated site recordings
    entry_mean = np.mean(all_ins_entry, axis=0)
    exit_mean = np.mean(all_ins_exit, axis=0)
    ins_mean = np.r_[[entry_mean], [exit_mean]]
    # Only consider deviation in ML and AP directions for this analysis
    # entry_std = np.std(all_ins_entry, axis=0)
    # entry_std[2]=0
    # exit_std = np.std(all_ins_exit, axis=0)
    # exit_std[2]=0
    # ins_upper = np.r_[[entry_mean+entry_std], [exit_mean+exit_std]]
    # ins_lower = np.r_[[entry_mean-entry_std], [exit_mean-exit_std]]
    
    # Plot the average track across all repeated site recordings, in RED
    cax.plot(ins_mean[:, 0] * 1e6, ins_mean[:, 2] * 1e6, 'orangered', linewidth=2)
    sax.plot(ins_mean[:, 1] * 1e6, ins_mean[:, 2] * 1e6, 'orangered', linewidth=2)
    hax.plot(ins_mean[0, 1] * 1e6, ins_mean[0, 0] * 1e6, 'orangered', marker="o",
             markersize=2)
    hax2.plot(ins_mean[1, 1] * 1e6, ins_mean[1, 0] * 1e6, 'orangered', marker="o",
              markersize=2)
    #hax2.plot(ins_plan.xyz[1, 1] * 1e6, ins_plan.xyz[1, 0] * 1e6, color='k',
    #          marker="o", markersize=6)
    
    # add planned insertion ON TOP of the actual insertions, in YELLOW
    cax.plot(ins_plan.xyz[:, 0] * 1e6, ins_plan.xyz[:, 2] * 1e6, 'gold', linewidth=2)
    sax.plot(ins_plan.xyz[:, 1] * 1e6, ins_plan.xyz[:, 2] * 1e6, 'gold', linewidth=2)
    hax.plot(ins_plan.xyz[0, 1] * 1e6, ins_plan.xyz[0, 0] * 1e6, color='gold',
             marker="o", markersize=2)
    hax2.plot(ins_plan.xyz[1, 1] * 1e6, ins_plan.xyz[1, 0] * 1e6, color='gold',
             marker="o", markersize=2)
    
    
    # Compute targeting error at surface of brain
    error_top = all_ins_entry - ins_plan.xyz[0, :]
    norm_top = np.sqrt(np.sum(error_top ** 2, axis=1))
    top_mean = np.mean(norm_top)*1e6
    top_std = np.std(norm_top)*1e6
    
    # Compute targeting error at tip of probe
    error_bottom = all_ins_exit - ins_plan.xyz[1, :]
    norm_bottom = np.sqrt(np.sum(error_bottom ** 2, axis=1))
    bottom_mean = np.mean(norm_bottom)*1e6
    bottom_std = np.std(norm_bottom)*1e6
    
    # Add targeting errors to the title of figures
    ax1.xaxis.label.set_size(8)
    ax1.tick_params(axis='x', labelsize=7)
    ax1.yaxis.label.set_size(8)
    ax1.tick_params(axis='y', labelsize=7)
    #ax1.set_title('HISTOLOGY: coronal plane error \n ' +
    #              str(np.around(alpha_mean, 1)) + r'$\pm$' +
    #              str(np.around(alpha_std, 1)) + ' deg', fontsize=18)
    ax1.yaxis.set_major_locator(plt.MaxNLocator(4))
    
    ax2.xaxis.label.set_size(8)
    ax2.tick_params(axis='x', labelsize=7)
    ax2.yaxis.label.set_size(8)
    ax2.tick_params(axis='y', labelsize=7)
    #ax2.set_title('HISTOLOGY: sagittal plane error \n ' +
    #              str(np.around(beta_mean, 1)) + r'$\pm$' +
    #              str(np.around(beta_std, 1)) + ' deg', fontsize=18)
    ax2.yaxis.set_major_locator(plt.MaxNLocator(4))
    
    ax3.set_xlabel('ap (um)', fontsize=8)
    ax3.tick_params(axis='x', labelsize=7)
    ax3.set_ylabel('ml (um)', fontsize=8)
    ax3.tick_params(axis='y', labelsize=7)
    ax3.set_title('HISTOLOGY: surface error \n ' +
                  str(np.around(top_mean, 1)) + r'$\pm$' +
                  str(np.around(top_std, 1)) + ' µm', fontsize=18)
    ax3.yaxis.set_major_locator(plt.MaxNLocator(4))
    
    ax4.set_xlabel('ap (µm)', fontsize=8)
    ax4.tick_params(axis='x', labelsize=7)
    ax4.set_ylabel('ml (µm)', fontsize=8)
    ax4.tick_params(axis='y', labelsize=7)
    ax4.set_title('HISTOLOGY: probe tip error \n ' +
                  str(np.around(bottom_mean, 1)) + r'$\pm$' +
                  str(np.around(bottom_std, 1)) + ' µm', fontsize=18)
    ax4.yaxis.set_major_locator(plt.MaxNLocator(4))
    
    ax1.set_ylim((-5000,0))
    ax1.set_xlim((0,-4000))
    ax2.set_ylim((-5000,0))
    ax2.set_xlim((0,-4000))
    ax3.set_ylim((-4000,0))
    ax3.set_xlim((0,-4000))
    ax4.set_ylim((-4000,0))
    ax4.set_xlim((0,-4000))
    
    


if __name__ == "__main__":
    plot_trajs()



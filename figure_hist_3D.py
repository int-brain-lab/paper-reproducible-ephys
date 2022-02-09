#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 14:32:00 2022

Create 3D rendering of trajectories used for repeated site trajectory analysis.

@author: sjwest
"""



def create_3d_traj_render(output = 'figure_histology_supp'):
    
    from pathlib import Path
    import os
    import figure_hist_data as fhd
    import matplotlib.pyplot as plt
    from one.api import ONE
    import numpy as np
    import ibllib.atlas as atlas
    import reproducible_ephys_functions as ref
    
    from mayavi import mlab
    from atlaselectrophysiology import rendering
    from ibllib.plots import color_cycle
    
    # output DIR
    OUTPUT = Path(output)
    
    # generate output DIR for storing plots:
    if os.path.exists(OUTPUT) is False:
        os.mkdir(OUTPUT)
    
    
    one = ONE()
    ba = atlas.AllenAtlas(25)
    traj_rep = one.alyx.rest('trajectories', 'list', provenance='Planned',
                             x=-2243, y=-2000,  project='ibl_neuropixel_brainwide_01')
    
    
    fig = rendering.figure(grid=False)
    
    # use repo-ephys figure style
    ref.figure_style()
    
    # get probe data if necessary
    if 'probe_data' not in vars() or 'probe_data' not in globals():
        # get probe_data histology query
        probe_data = fhd.get_probe_data()
    
    traj_data = probe_data
    
    # output DIR - generate output DIR for storing plots:
    #OUTPUT = Path(output)
    #if os.path.exists(OUTPUT) is False:
    #    os.mkdir(OUTPUT)
    
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
        
        mlapdv = ba.xyz2ccf(ins.xyz)
        # display the trajectories
        color = color_cycle(1)
        color = tuple(institution_colors[institution_map[phi_lab]])
        mlab.plot3d(mlapdv[:, 1], mlapdv[:, 2], mlapdv[:, 0],
                    line_width=1, tube_radius=20, color=color)
    
    
    # add planned ins?
    #mlapdv = ba.xyz2ccf(ins_plan.xyz)
    #color = tuple(institution_colors[institution_map[phi_lab]])
    #mlab.plot3d(mlapdv[:, 1], mlapdv[:, 2], mlapdv[:, 0],
    #            line_width=5, tube_radius=50, color=color)
    
    
    rendering.rotating_video( str(Path(OUTPUT, 'fig_hist_probe_placement_3D.webm')), fig)
    



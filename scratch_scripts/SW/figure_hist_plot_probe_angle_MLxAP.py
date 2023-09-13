#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 13:33:54 2021

This Python Script generates a figure plotting the PLANNED repeated site
insertion coordinate at the brain surface at [0,0], and then the VECTORS from 
planned surface to actual surface coord of histology tracks.  The points
of the histology track surface coords are coloured based on lab affiliation.

@author: sjwest
"""

def print_path():
    import os
    path = os.path.dirname(os.path.realpath(__file__))
    print(path)




def plot_probe_angle_histology_panel(output = 'figure_histology'):
    '''
    Plot the whole probe histology panel, consisting of: 
    
    * scatterplot of the PLANNED to HISTOLOGY angles at brain surface, 
    
    * horizontal boxplot plus distplot (density plot) of all PLANNED to 
    HISTOLOGY angle values (to see total distribution),
    
    * horizontal boxplots of each labs distribution
    
    * heat map of each labs permutation test p-value.?
    
    Panel saved to output as: angle_histology_panel.svg

    Returns
    -------
    None.

    '''
    from pathlib import Path
    import os
    import figure_hist_data as fhd
    import svgutils.compose as sc # layout figure in svgutils
    
    # output DIR
    OUTPUT = Path(output)
    if os.path.exists(OUTPUT) is False:
        os.mkdir(OUTPUT) # generate output DIR for storing plots:
    
    # get probe_data histology query
    probe_data = fhd.get_probe_data()
    
    
    # generate scatterplot in first axes
    plot_probe_angle_histology(probe_data, output=output) # saves as SVG to output
    
    # generate histogram/density plot of Euclidean distance at surface from 
    # planned to actual for all trajectories
    # AND plot by lab
    plot_probe_angle_histology_all_lab(probe_data, output=output)
    
    # generate grouped boxplots of Euclidean distance at surface from 
    # planned to actual for all trajectories, by LAB
    #plot_probe_angle_histology_lab(probe_data, output=output)
    
    
    # generate horizontal boxplot plus density plot of all surf coord histology
     # in second axes
    #plot_probe_distance_histology_all(probe_data, ax2)
    fig = sc.Figure( "66mm", "140mm",
        
        sc.Panel(
            sc.SVG(output+os.path.sep+'E_probe_angle_hist_label.svg'
                   ).scale(0.35)
            ),
        
        sc.Panel(
            sc.SVG(output+os.path.sep+'E_probe_angle_hist_all_lab.svg'
                ).scale(0.35
                ).move(0,68)
            ),
        
        #sc.Grid(20, 20)
        )
    
    fig.save( Path(output, "angle_histology_panel.svg") )
    




def plot_probe_angle_histology(probe_data, output='figure_histology'):
    '''Plot the PLANNED probe angle at [0,0], VECTORS from planned angle to
    actual angle of histology tracks, histology track points coloured
    by lab affiliation.
    '''
    from pathlib import Path
    import os
    import figure_hist_data as fhd
    import matplotlib.pyplot as plt
    from one.api import ONE
    from iblatlas.atlas import Insertion
    import numpy as np
    import atlaselectrophysiology.load_histology as hist
    import iblatlas.atlas as atlas
    import reproducible_ephys_functions as ref
    import math
    import statistics as stat
    
    # set fig style to repo-ephys
    ref.figure_style()
    
    # get probe data if necessary
    if 'probe_data' not in vars() or 'probe_data' not in globals():
        # get probe_data histology query
        probe_data = fhd.get_probe_data()
    
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
    ins_plan_traj = ins_plan.trajectory
    
    # get new atlas for plotting
    #brain_atlas = atlas.AllenAtlas(res_um=25)
    
    fig1, ax1 = plt.subplots()
    
    # empty numpy arrays for storing the entry point of probe into brain
     # and "exit point" i.e the probe tip!
    all_ins_entry = np.empty((0, 3))
    
    # get institution map and colours
    lab_number_map, institution_map, institution_colors = ref.labs()
    
    # ins_plan - insertion for planned
    ml_list = list()
    ap_list = list()
    lab_list = list()
    angle_list = list()
    
    # plot line and points from planned insertion entry to actual histology entry
    #for idx in range(len(probe_data)):
    for idx, row in probe_data.iterrows():
        
        phi_lab = row['lab']
        
        lab_list.append(phi_lab)
        
        # get the insertion object for histology
        ins_hist = Insertion.from_dict({ 'x': float(row['hist_x']), 
                             'y': float(row['hist_y']), 
                             'z': float(row['hist_z']), 
                             'phi': float(row['hist_phi']), 
                             'theta': float(row['hist_theta']), 
                             'depth': float(row['hist_depth']) })
        
        # compute the difference in entry coord between ins_plan and ins_hist
        # and use this to move the ins_hist TIP coord
        # ins_plan.xyz[0, :] surface and ins_plan.xyz[1, :] tip
        ins_diff = ins_plan.xyz[0,:] - ins_hist.xyz[0,:] # get difference between surface insertions
        # ins_hist.xyz[0,:] + ins_diff # GIVES ins_plan.xyz[0,:] !!
         # i.e it MOVES the surface coord of ins_hist to ins_plan surface coord!
        hist_tip_norm = ins_hist.xyz[1,:] + ins_diff # gives tip coord, with surface moved to planned ins coord
        # ins_plan.xyz[1,:] ins_hist[1,:] - coords are ML, AP, DV
          # want to IGNORE DV - look at direction between coords in ML, AP
        
        # calculate the ANGLE between ins_plan TIP -> SURFACE -> hist_tip_norm
         # ins_plan.xyz[1, :] -> ins_plan.xyz[0, :] -> hist_tip_norm
        # from: https://stackoverflow.com/questions/35176451/python-code-to-calculate-angle-between-three-point-using-their-3d-coordinates
        ba = ins_plan.xyz[1, :] - ins_plan.xyz[0, :]
        bc = hist_tip_norm - ins_plan.xyz[0, :]
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        
        angle = np.arccos(cosine_angle) # returns angle in RADIANS
        angle_deg = np.degrees(angle) # can use degrees if desired
        
        angle_list.append(angle_deg)
        # want to PLOT the MAGNITUDE of this angle!
        
        # NEXT compute the DIRECTION of the angle from planned to hist
         # get this by first computing the normal vector from hist_tip_norm 
         # to the planned trajectory
    
        # project hist_tip_norm to the PLANNED trajectory:
        hist_tip_norm_vector = ins_plan_traj.project(hist_tip_norm)
        
        # this gives the coord on ins_plan_traj which makes a right angle with
         # hist_tip_norm - so together these coords give the NORMAL VECTOR to
         # ins_plan_traj
        
        # can also  calculate the distance between hist_tip_norm and ins_plan trajectory:
        hist_tip_dist = np.linalg.norm( hist_tip_norm - hist_tip_norm_vector ) *1e6
        
        # from the NORMAL VECTOR can comptue the ML and AP distances by simply subtracting
        # the two values in the vector
        ML_dist = hist_tip_norm[0] - hist_tip_norm_vector[0]
        AP_dist = hist_tip_norm[1] - hist_tip_norm_vector[1]
        
        # and thus can compute the ABSOLUTE RATIO that the ANGLE MAGNITUDE is shared between
        # ML and AP
        ML_ratio = abs(ML_dist) / (abs(ML_dist) + abs(AP_dist))
        AP_ratio = abs(AP_dist) / (abs(ML_dist) + abs(AP_dist))
        
        # combining this ratio with the SIGN of ML_dist/AP_dist can be used to compute the coord
        # of the ANGLE MAGNITUDE from 0,0 as the planned trajectory
        
        # using pythagoras - compute the distance of the hypotenuse 
        # if using ML_ratio and AP_ratio as the right angle lengths
        hyp = math.sqrt( (ML_ratio*ML_ratio) + (AP_ratio*AP_ratio) )
        
        # use this to calculate the proportions of each ratio
        angle_ML = (angle_deg / hyp) * ML_ratio
        angle_AP = (angle_deg / hyp) * AP_ratio
        
        # confirm this works by checking the total length is correct with pythagoras
        #math.sqrt( (angle_ML*angle_ML) + (angle_AP*angle_AP))
        
        # finally, flip the sign depending on if ML/AP_dist is POSITIVE or NEGATIVE
        if ML_dist < 0:
            angle_ML = -(angle_ML)
        
        if AP_dist < 0:
            angle_AP = -(angle_AP)
        
        # add values to lists
        ml_list.append(angle_ML)
        ap_list.append(angle_AP)
        
    
    # now generate plot - loop through points and plot
    
    # draw 0,0 lines
    ax1.axhline(y=0, color="grey", linestyle="--", linewidth = 0.5)
    ax1.axvline(x=0, color="grey", linestyle="--", linewidth = 0.5)
    
    # ALL LINES
    for idx, m in enumerate(ml_list):
                
        # PLOT the line at [angle_AP, angle_ML] [x, y]
        ax1.plot( (ml_list[idx], 0), (ap_list[idx],0), 
             color= institution_colors[institution_map[lab_list[idx]]], 
             linewidth = 0.15, alpha = 0.8 )
    
    # THEN ALL POINTS
    for idx, m in enumerate(ml_list):
                
        # PLOT the point at [angle_ML, angle_AP]
        ax1.plot(ml_list[idx], ap_list[idx], 
             color= institution_colors[institution_map[lab_list[idx]]], 
              marker="o", markersize=0.5, alpha = 0.8, markeredgewidth = 0.5 )
    
    # add institution col
    probe_data['inst'] = probe_data['lab'].map(institution_map)
    
    
    probe_data['ml_angle'] = ml_list
    probe_data['ap_angle'] = ap_list
    probe_data['angle'] = angle_list
    lab_mean_histml = probe_data.groupby('lab')['ml_angle'].mean()
    lab_mean_histap = probe_data.groupby('lab')['ap_angle'].mean()
    
    for ml, ap, k in zip(lab_mean_histml, lab_mean_histap, lab_mean_histml.keys()):
        ax1.plot( [ ml ], 
                  [ ap ], 
              color= institution_colors[institution_map[k]], 
              marker="+", markersize=3, alpha = 0.5,
              label = institution_map[k])
    
    # plot the mean histology coords
    mean_histml = stat.mean(ml_list)
    mean_histap = stat.mean(ap_list)
    
    ax1.plot( [ mean_histml ], 
                  [ mean_histap ], 
              color= 'k', marker="+", markersize=6, alpha = 0.7,
              label = "MEAN")
    
    # set x/y axis labels
    ax1.set_xlabel('histology ML angle (degrees)', fontsize=6)
    #ax1.tick_params(axis='x', labelsize=7)
    ax1.set_ylabel('histology AP angle (degrees)', fontsize=6)
    #ax1.tick_params(axis='y', labelsize=7)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(7))
    ax1.yaxis.set_major_locator(plt.MaxNLocator(7))
    
    #ax1.set_ylim((-3000,-1250))
    #ax1.set_xlim((-1000,-3500))
    #ax1.set_ylim((-3000,-500))
    #ax1.set_xlim((-500,-3500))
    ml_length = max([max(ml_list), abs(min(ml_list))])
    ap_length = max([max(ap_list), abs(min(ap_list))])
    ml_length = round( (ml_length+0.5) )
    ap_length = round( (ap_length+0.5) )
    #ax1.set_ylim((-(ml_length),ml_length))
    #ax1.set_xlim((-(ap_length),ap_length))
    ax1.set_ylim((-(20),10))
    ax1.set_xlim((-(20),10))
    # or just to min and max
    #ax1.set_xlim((min(ml_list),max(ml_list)))
    #ax1.set_ylim((min(ap_list),max(ap_list)))
    
    
    # Move left y-axis and bottim x-axis to centre, passing through (0,0)
    #ax1.spines['left'].set_position('center')
    #ax1.spines['bottom'].set_position('center')
    # Eliminate upper and right axes
    #ax1.spines['right'].set_color('none')
    #ax1.spines['top'].set_color('none')
    #ax1.spines['right'].set_position('center')
    #ax1.spines['top'].set_position('center')
    
    plt.tight_layout() # tighten layout around xlabel & ylabel
    
    fig1.set_size_inches(2.15, 2.15)
    
    # add a subplot INSIDE the fig1 ax1
    axav = fig1.add_axes([0.1,0.12,0.28,0.28])
    
    #axav.axes.xaxis.set_ticks([])
    #axav.axes.yaxis.set_ticks([])
    axav.xaxis.tick_top()
    axav.yaxis.tick_right()
    axav.tick_params(axis='both', labelsize=3, pad = 1)
    #axav.xaxis.labelpad = 1
    #axav.yaxis.labelpad = 1
    
    axav.axhline(y=0, color="grey", linestyle="--", linewidth = 0.5)
    axav.axvline(x=0, color="grey", linestyle="--", linewidth = 0.5)
    axav.set_xlim((-10,5))
    axav.set_ylim((-10,5))
    
    for ml, ap, k in zip(lab_mean_histml, lab_mean_histap, lab_mean_histml.keys()):
        axav.plot( [ ml ], 
                  [ ap ], 
              color= institution_colors[institution_map[k]], 
              marker="+", markersize=5, alpha = 0.7,
              label = institution_map[k])
    
    # plot the mean histology coords
    mean_histml = stat.mean(ml_list)
    mean_histap = stat.mean(ap_list)
    
    axav.plot( [ mean_histml ], 
                  [ mean_histap ], 
              color= 'k', marker="+", markersize=8, alpha = 0.7,
              label = "MEAN")
    
    fig1.savefig( str(Path(OUTPUT, 'E_probe_angle_hist.svg')), bbox_inches="tight" )
    #fig1.savefig( str(Path('figure_histology', 'probe-plots','micro_surf_err_plot.svg')), bbox_inches="tight" ) # tight ensures figure is in bounds of svg canvas!
    
    # create new column to indicate if each row passes advanced query
    passed_eids = ref.eid_list()
    # permutation testing - PASS DATA ONLY
    probe_data_passed = probe_data[probe_data.eid.isin(passed_eids)]
    
    # add mean trageting error distance to title
    ax1.set_title('Mean (SD) angle \n' +
                  'ALL: ' + str(np.around( np.mean(angle_list), 1)) + 
                  ' ('+str(np.around(np.std(angle_list), 2))+')'+ ' degrees\n' + 
                  'PASSED:' + str(np.around( np.mean(probe_data_passed['angle']), 1)) + 
                  ' ('+str(np.around(np.std(probe_data_passed['angle']), 2))+')'+ 
                  ' degrees', fontsize=8)
    
    fig1.savefig( str(Path(OUTPUT, 'E_probe_angle_hist_label.svg')), bbox_inches="tight" )
    #fig1.savefig( str(Path('figure_histology', 'probe-plots','micromanipulator_surface_error_plot.svg')), bbox_inches="tight" ) # tight ensures figure is in bounds of svg canvas!
    



def plot_probe_angle_histology_all_lab(probe_data, output='figure_histology'):
    '''Plot the DISTANCES from planned to histology angles, histology track 
    boxplot of ALL angles - to see its distribution shape.
    '''
    from pathlib import Path
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    from one.api import ONE
    from iblatlas.atlas import Insertion
    import numpy as np
    import atlaselectrophysiology.load_histology as hist
    import iblatlas.atlas as atlas
    import reproducible_ephys_functions as ref
    import figure_hist_data as fhd
    import math
    import statistics as stat
    
    # set fig style to repo-ephys
    ref.figure_style()
    
    # get probe data if necessary
    if 'probe_data' not in vars() or 'probe_data' not in globals():
        # get probe_data histology query
        probe_data = fhd.get_probe_data()
    
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
    ins_plan_traj = ins_plan.trajectory
    
    # empty numpy arrays for storing the entry point of probe into brain
     # and "exit point" i.e the probe tip!
    all_ins_entry = np.empty((0, 3))
    
    # get institution map and colours
    lab_number_map, institution_map, institution_colors = ref.labs()
    
    # ins_plan - insertion for planned
    ml_list = list()
    ap_list = list()
    lab_list = list()
    angle_list = list()
    
    # plot line and points from planned insertion entry to actual histology entry
    #for idx in range(len(probe_data)):
    for idx, row in probe_data.iterrows():
        
        phi_lab = row['lab']
        
        lab_list.append(phi_lab)
        
        # get the insertion object for histology
        ins_hist = Insertion.from_dict({ 'x': float(row['hist_x']), 
                             'y': float(row['hist_y']), 
                             'z': float(row['hist_z']), 
                             'phi': float(row['hist_phi']), 
                             'theta': float(row['hist_theta']), 
                             'depth': float(row['hist_depth']) })
        
        # compute the difference in entry coord between ins_plan and ins_hist
        # and use this to move the ins_hist TIP coord
        # ins_plan.xyz[0, :] surface and ins_plan.xyz[1, :] tip
        ins_diff = ins_plan.xyz[0,:] - ins_hist.xyz[0,:] # get difference between surface insertions
        # ins_hist.xyz[0,:] + ins_diff # GIVES ins_plan.xyz[0,:] !!
         # i.e it MOVES the surface coord of ins_hist to ins_plan surface coord!
        hist_tip_norm = ins_hist.xyz[1,:] + ins_diff # gives tip coord, with surface moved to planned ins coord
        # ins_plan.xyz[1,:] ins_hist[1,:] - coords are ML, AP, DV
          # want to IGNORE DV - look at direction between coords in ML, AP
        
        # calculate the ANGLE between ins_plan TIP -> SURFACE -> hist_tip_norm
         # ins_plan.xyz[1, :] -> ins_plan.xyz[0, :] -> hist_tip_norm
        # from: https://stackoverflow.com/questions/35176451/python-code-to-calculate-angle-between-three-point-using-their-3d-coordinates
        ba = ins_plan.xyz[1, :] - ins_plan.xyz[0, :]
        bc = hist_tip_norm - ins_plan.xyz[0, :]
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        
        angle = np.arccos(cosine_angle) # returns angle in RADIANS
        angle_deg = np.degrees(angle) # can use degrees if desired
        
        angle_list.append(angle_deg)
        # want to PLOT the MAGNITUDE of this angle!
        
        # NEXT compute the DIRECTION of the angle from planned to hist
         # get this by first computing the normal vector from hist_tip_norm 
         # to the planned trajectory
    
        # project hist_tip_norm to the PLANNED trajectory:
        hist_tip_norm_vector = ins_plan_traj.project(hist_tip_norm)
        
        # this gives the coord on ins_plan_traj which makes a right angle with
         # hist_tip_norm - so together these coords give the NORMAL VECTOR to
         # ins_plan_traj
        
        # can also  calculate the distance between hist_tip_norm and ins_plan trajectory:
        hist_tip_dist = np.linalg.norm( hist_tip_norm - hist_tip_norm_vector ) *1e6
        
        # from the NORMAL VECTOR can comptue the ML and AP distances by simply subtracting
        # the two values in the vector
        ML_dist = hist_tip_norm[0] - hist_tip_norm_vector[0]
        AP_dist = hist_tip_norm[1] - hist_tip_norm_vector[1]
        
        # and thus can compute the ABSOLUTE RATIO that the ANGLE MAGNITUDE is shared between
        # ML and AP
        ML_ratio = abs(ML_dist) / (abs(ML_dist) + abs(AP_dist))
        AP_ratio = abs(AP_dist) / (abs(ML_dist) + abs(AP_dist))
        
        # combining this ratio with the SIGN of ML_dist/AP_dist can be used to compute the coord
        # of the ANGLE MAGNITUDE from 0,0 as the planned trajectory
        
        # using pythagoras - compute the distance of the hypotenuse 
        # if using ML_ratio and AP_ratio as the right angle lengths
        hyp = math.sqrt( (ML_ratio*ML_ratio) + (AP_ratio*AP_ratio) )
        
        # use this to calculate the proportions of each ratio
        angle_ML = (angle_deg / hyp) * ML_ratio
        angle_AP = (angle_deg / hyp) * AP_ratio
        
        # confirm this works by checking the total length is correct with pythagoras
        #math.sqrt( (angle_ML*angle_ML) + (angle_AP*angle_AP))
        
        # finally, flip the sign depending on if ML/AP_dist is POSITIVE or NEGATIVE
        if ML_dist < 0:
            angle_ML = -(angle_ML)
        
        if AP_dist < 0:
            angle_AP = -(angle_AP)
        
        # add values to lists
        ml_list.append(angle_ML)
        ap_list.append(angle_AP)
        
    
    # now generate plot - loop through points and plot
    
    # add institution col
    probe_data['inst'] = probe_data['lab'].map(institution_map)
    
    
    probe_data['ml_angle'] = ml_list
    probe_data['ap_angle'] = ap_list
    probe_data['angle'] = angle_list
    lab_mean_histml = probe_data.groupby('lab')['ml_angle'].mean()
    lab_mean_histap = probe_data.groupby('lab')['ap_angle'].mean()
    
    # create boxplot with institution colours
    #sns.boxplot(y='angle', data=probe_data, 
    #            color = 'white', 
    #            ax=ax1)
    #sns.stripplot(y='angle', data=probe_data, 
    #              color = 'black', alpha = 0.8, size = 3, 
    #              ax=ax1)
    
    # create new column to indicate if each row passes advanced query
    passed_eids = ref.eid_list()
    passed_adv = []
    for p in probe_data['eid']:
        if p in passed_eids:
            passed_adv.append("PASS")
        else:
            passed_adv.append("FAIL")
    
    probe_data['passed_adv'] = passed_adv
    
    # Create an array with the colors you want to use
    colors = ["#000000", "#FF0B04"]
    # Set your custom color palette
    sns.set_palette(sns.color_palette(colors))
    
    fig, (ax1, ax2) = plt.subplots(2, gridspec_kw={'height_ratios': [1, 2]})
    
    sns.histplot(probe_data['angle'], kde=True, 
                 color = 'grey',
                 ax = ax1)
    
    ax1.set_xlim(0, 20)
    #ax1.set_xlabel('Institution', fontsize=7)
    ax1.set_ylabel('count')
    #ax1.set_xlabel('Histology distance (µm)')
    ax1.set_xlabel(None)
    #ax1.get_legend().remove()
    #ax1.tick_params(axis='x', labelrotation = 90)
    
    ax1.xaxis.set_major_locator(plt.MaxNLocator(5))
    #ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90, ha='right')
    ax1.tick_params(axis='x', labelrotation=90)
    ax1.set(xticklabels=[])
    ax1.tick_params(bottom=False)
    #ax1.set_axis_off()
    
    sns.stripplot(y='inst', x='angle', data=probe_data, 
                  hue = 'passed_adv',
                  alpha = 0.8, size = 1.5, 
                   orient="h",
                  ax=ax2)
    
    # plot the mean line
    sns.boxplot(showmeans=True,
            meanline=True,
            meanprops={'color': 'gray', 'ls': '-', 'lw': 1},
            medianprops={'visible': False},
            whiskerprops={'visible': False},
            zorder=10,
            x="angle",
            y="inst",
            data=probe_data,
            showfliers=False,
            showbox=False,
            showcaps=False,
            ax=ax2)
    
    #ax1.set_xlabel('Institution', fontsize=7)
    ax2.set_ylabel(None)
    ax2.set_xlim(0, 20)
    ax2.set_xlabel('Histology angle (degrees)')
    
    #ax1.tick_params(axis='x', labelrotation = 90)
    
    ax2.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax2.tick_params(axis='x', labelrotation=90)
    ax2.get_legend().remove()
    
    p_hist1 = permutation_test_histology()
    p_hist2 = permutation_test_histology()
    p_hist3 = permutation_test_histology()
    p_hist = stat.mean([p_hist1, p_hist2, p_hist3])
    print("Histology PERMUTATION TEST ALL : ", p_hist)
    
    # permutation test on PASS eids ONLY
    passed_eids = ref.eid_list()
    pp_h1 = permutation_test_histology(passed_eids)
    pp_h2 = permutation_test_histology(passed_eids)
    pp_h3 = permutation_test_histology(passed_eids)
    pp_h = stat.mean([pp_h1, pp_h2, pp_h3])
    print("Histology PERMUTATION TEST PASS : ", pp_h)
    
    ax1.set_title('Permutation Test p-value: \n    ALL : ' 
                    + str( round( p_hist, 4) )
                    + '    PASS : ' + str( round( pp_h, 4) ) )
    
    plt.tight_layout() # tighten layout around xlabel & ylabel
    
    fig.set_size_inches(2.15, 2.8)
    fig.savefig( str(Path(OUTPUT, 'E_probe_angle_hist_all_lab.svg')), bbox_inches="tight" )
    #fig1.savefig( str(Path('figure_histology', 'probe-plots','micro_surf_err_plot.svg')), bbox_inches="tight" ) # tight ensures figure is in bounds of svg canvas!
    




def plot_probe_angle_histology_all(probe_data, output='figure_histology'):
    '''Plot the DISTANCES from planned to histology angles, histology track 
    boxplot of ALL angles - to see its distribution shape.
    '''
    from pathlib import Path
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    from one.api import ONE
    from iblatlas.atlas import Insertion
    import numpy as np
    import atlaselectrophysiology.load_histology as hist
    import iblatlas.atlas as atlas
    import reproducible_ephys_functions as ref
    import figure_hist_data as fhd
    import math
    
    # set fig style to repo-ephys
    ref.figure_style()
    
    # get probe data if necessary
    if 'probe_data' not in vars() or 'probe_data' not in globals():
        # get probe_data histology query
        probe_data = fhd.get_probe_data()
    
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
    ins_plan_traj = ins_plan.trajectory
    
    # empty numpy arrays for storing the entry point of probe into brain
     # and "exit point" i.e the probe tip!
    all_ins_entry = np.empty((0, 3))
    
    # get institution map and colours
    lab_number_map, institution_map, institution_colors = ref.labs()
    
    # ins_plan - insertion for planned
    ml_list = list()
    ap_list = list()
    lab_list = list()
    angle_list = list()
    
    # plot line and points from planned insertion entry to actual histology entry
    #for idx in range(len(probe_data)):
    for idx, row in probe_data.iterrows():
        
        phi_lab = row['lab']
        
        lab_list.append(phi_lab)
        
        # get the insertion object for histology
        ins_hist = Insertion.from_dict({ 'x': float(row['hist_x']), 
                             'y': float(row['hist_y']), 
                             'z': float(row['hist_z']), 
                             'phi': float(row['hist_phi']), 
                             'theta': float(row['hist_theta']), 
                             'depth': float(row['hist_depth']) })
        
        # compute the difference in entry coord between ins_plan and ins_hist
        # and use this to move the ins_hist TIP coord
        # ins_plan.xyz[0, :] surface and ins_plan.xyz[1, :] tip
        ins_diff = ins_plan.xyz[0,:] - ins_hist.xyz[0,:] # get difference between surface insertions
        # ins_hist.xyz[0,:] + ins_diff # GIVES ins_plan.xyz[0,:] !!
         # i.e it MOVES the surface coord of ins_hist to ins_plan surface coord!
        hist_tip_norm = ins_hist.xyz[1,:] + ins_diff # gives tip coord, with surface moved to planned ins coord
        # ins_plan.xyz[1,:] ins_hist[1,:] - coords are ML, AP, DV
          # want to IGNORE DV - look at direction between coords in ML, AP
        
        # calculate the ANGLE between ins_plan TIP -> SURFACE -> hist_tip_norm
         # ins_plan.xyz[1, :] -> ins_plan.xyz[0, :] -> hist_tip_norm
        # from: https://stackoverflow.com/questions/35176451/python-code-to-calculate-angle-between-three-point-using-their-3d-coordinates
        ba = ins_plan.xyz[1, :] - ins_plan.xyz[0, :]
        bc = hist_tip_norm - ins_plan.xyz[0, :]
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        
        angle = np.arccos(cosine_angle) # returns angle in RADIANS
        angle_deg = np.degrees(angle) # can use degrees if desired
        
        angle_list.append(angle_deg)
        # want to PLOT the MAGNITUDE of this angle!
        
        # NEXT compute the DIRECTION of the angle from planned to hist
         # get this by first computing the normal vector from hist_tip_norm 
         # to the planned trajectory
    
        # project hist_tip_norm to the PLANNED trajectory:
        hist_tip_norm_vector = ins_plan_traj.project(hist_tip_norm)
        
        # this gives the coord on ins_plan_traj which makes a right angle with
         # hist_tip_norm - so together these coords give the NORMAL VECTOR to
         # ins_plan_traj
        
        # can also  calculate the distance between hist_tip_norm and ins_plan trajectory:
        hist_tip_dist = np.linalg.norm( hist_tip_norm - hist_tip_norm_vector ) *1e6
        
        # from the NORMAL VECTOR can comptue the ML and AP distances by simply subtracting
        # the two values in the vector
        ML_dist = hist_tip_norm[0] - hist_tip_norm_vector[0]
        AP_dist = hist_tip_norm[1] - hist_tip_norm_vector[1]
        
        # and thus can compute the ABSOLUTE RATIO that the ANGLE MAGNITUDE is shared between
        # ML and AP
        ML_ratio = abs(ML_dist) / (abs(ML_dist) + abs(AP_dist))
        AP_ratio = abs(AP_dist) / (abs(ML_dist) + abs(AP_dist))
        
        # combining this ratio with the SIGN of ML_dist/AP_dist can be used to compute the coord
        # of the ANGLE MAGNITUDE from 0,0 as the planned trajectory
        
        # using pythagoras - compute the distance of the hypotenuse 
        # if using ML_ratio and AP_ratio as the right angle lengths
        hyp = math.sqrt( (ML_ratio*ML_ratio) + (AP_ratio*AP_ratio) )
        
        # use this to calculate the proportions of each ratio
        angle_ML = (angle_deg / hyp) * ML_ratio
        angle_AP = (angle_deg / hyp) * AP_ratio
        
        # confirm this works by checking the total length is correct with pythagoras
        #math.sqrt( (angle_ML*angle_ML) + (angle_AP*angle_AP))
        
        # finally, flip the sign depending on if ML/AP_dist is POSITIVE or NEGATIVE
        if ML_dist < 0:
            angle_ML = -(angle_ML)
        
        if AP_dist < 0:
            angle_AP = -(angle_AP)
        
        # add values to lists
        ml_list.append(angle_ML)
        ap_list.append(angle_AP)
        
    
    # now generate plot - loop through points and plot
    
    # add institution col
    probe_data['inst'] = probe_data['lab'].map(institution_map)
    
    
    probe_data['ml_angle'] = ml_list
    probe_data['ap_angle'] = ap_list
    probe_data['angle'] = angle_list
    lab_mean_histml = probe_data.groupby('lab')['ml_angle'].mean()
    lab_mean_histap = probe_data.groupby('lab')['ap_angle'].mean()
    
    # create boxplot with institution colours
    #sns.boxplot(y='angle', data=probe_data, 
    #            color = 'white', 
    #            ax=ax1)
    #sns.stripplot(y='angle', data=probe_data, 
    #              color = 'black', alpha = 0.8, size = 3, 
    #              ax=ax1)
    
    fig = sns.displot(probe_data['angle'], kde=True)
    fig1 = fig.fig # fig is a FacetGrid object!
    ax1 = fig.ax # get axes
    
    #ax1.set_xlabel('Institution', fontsize=7)
    #ax1.set_xlabel('Histology angle (degrees)')
    #ax1.set_ylabel('Histology distance (µm)', fontsize=7)
    ax1.set_ylabel(None)
    ax1.set_xlabel(None)
    
    ax1.xaxis.set_major_locator(plt.MaxNLocator(5))
    #ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90, ha='right')
    ax1.tick_params(axis='x', labelrotation=90)
    
    plt.tight_layout() # tighten layout around xlabel & ylabel
    
    fig1.set_size_inches(2.15, 1)
    fig1.savefig( str(Path(OUTPUT, 'E_probe_angle_hist_all.svg')), bbox_inches="tight" )
    #fig1.savefig( str(Path('figure_histology', 'probe-plots','micro_surf_err_plot.svg')), bbox_inches="tight" ) # tight ensures figure is in bounds of svg canvas!
    




def plot_probe_angle_histology_lab(probe_data, output='figure_histology'):
    '''Plot the DISTANCES from planned to histology angles, histology track 
    boxplot of ALL angles - to see its distribution shape.
    '''
    from pathlib import Path
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    from one.api import ONE
    from iblatlas.atlas import Insertion
    import numpy as np
    import atlaselectrophysiology.load_histology as hist
    import iblatlas.atlas as atlas
    import reproducible_ephys_functions as ref
    import figure_hist_data as fhd
    import math
    import statistics as stat
    
    # set fig style to repo-ephys
    ref.figure_style()
    
    # get probe data if necessary
    if 'probe_data' not in vars() or 'probe_data' not in globals():
        # get probe_data histology query
        probe_data = fhd.get_probe_data()
    
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
    ins_plan_traj = ins_plan.trajectory
    
    # empty numpy arrays for storing the entry point of probe into brain
     # and "exit point" i.e the probe tip!
    all_ins_entry = np.empty((0, 3))
    
    # get institution map and colours
    lab_number_map, institution_map, institution_colors = ref.labs()
    
    # ins_plan - insertion for planned
    ml_list = list()
    ap_list = list()
    lab_list = list()
    angle_list = list()
    
    # plot line and points from planned insertion entry to actual histology entry
    #for idx in range(len(probe_data)):
    for idx, row in probe_data.iterrows():
        
        phi_lab = row['lab']
        
        lab_list.append(phi_lab)
        
        # get the insertion object for histology
        ins_hist = Insertion.from_dict({ 'x': float(row['hist_x']), 
                             'y': float(row['hist_y']), 
                             'z': float(row['hist_z']), 
                             'phi': float(row['hist_phi']), 
                             'theta': float(row['hist_theta']), 
                             'depth': float(row['hist_depth']) })
        
        # compute the difference in entry coord between ins_plan and ins_hist
        # and use this to move the ins_hist TIP coord
        # ins_plan.xyz[0, :] surface and ins_plan.xyz[1, :] tip
        ins_diff = ins_plan.xyz[0,:] - ins_hist.xyz[0,:] # get difference between surface insertions
        # ins_hist.xyz[0,:] + ins_diff # GIVES ins_plan.xyz[0,:] !!
         # i.e it MOVES the surface coord of ins_hist to ins_plan surface coord!
        hist_tip_norm = ins_hist.xyz[1,:] + ins_diff # gives tip coord, with surface moved to planned ins coord
        # ins_plan.xyz[1,:] ins_hist[1,:] - coords are ML, AP, DV
          # want to IGNORE DV - look at direction between coords in ML, AP
        
        # calculate the ANGLE between ins_plan TIP -> SURFACE -> hist_tip_norm
         # ins_plan.xyz[1, :] -> ins_plan.xyz[0, :] -> hist_tip_norm
        # from: https://stackoverflow.com/questions/35176451/python-code-to-calculate-angle-between-three-point-using-their-3d-coordinates
        ba = ins_plan.xyz[1, :] - ins_plan.xyz[0, :]
        bc = hist_tip_norm - ins_plan.xyz[0, :]
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        
        angle = np.arccos(cosine_angle) # returns angle in RADIANS
        angle_deg = np.degrees(angle) # can use degrees if desired
        
        angle_list.append(angle_deg)
        # want to PLOT the MAGNITUDE of this angle!
        
        # NEXT compute the DIRECTION of the angle from planned to hist
         # get this by first computing the normal vector from hist_tip_norm 
         # to the planned trajectory
    
        # project hist_tip_norm to the PLANNED trajectory:
        hist_tip_norm_vector = ins_plan_traj.project(hist_tip_norm)
        
        # this gives the coord on ins_plan_traj which makes a right angle with
         # hist_tip_norm - so together these coords give the NORMAL VECTOR to
         # ins_plan_traj
        
        # can also  calculate the distance between hist_tip_norm and ins_plan trajectory:
        hist_tip_dist = np.linalg.norm( hist_tip_norm - hist_tip_norm_vector ) *1e6
        
        # from the NORMAL VECTOR can comptue the ML and AP distances by simply subtracting
        # the two values in the vector
        ML_dist = hist_tip_norm[0] - hist_tip_norm_vector[0]
        AP_dist = hist_tip_norm[1] - hist_tip_norm_vector[1]
        
        # and thus can compute the ABSOLUTE RATIO that the ANGLE MAGNITUDE is shared between
        # ML and AP
        ML_ratio = abs(ML_dist) / (abs(ML_dist) + abs(AP_dist))
        AP_ratio = abs(AP_dist) / (abs(ML_dist) + abs(AP_dist))
        
        # combining this ratio with the SIGN of ML_dist/AP_dist can be used to compute the coord
        # of the ANGLE MAGNITUDE from 0,0 as the planned trajectory
        
        # using pythagoras - compute the distance of the hypotenuse 
        # if using ML_ratio and AP_ratio as the right angle lengths
        hyp = math.sqrt( (ML_ratio*ML_ratio) + (AP_ratio*AP_ratio) )
        
        # use this to calculate the proportions of each ratio
        angle_ML = (angle_deg / hyp) * ML_ratio
        angle_AP = (angle_deg / hyp) * AP_ratio
        
        # confirm this works by checking the total length is correct with pythagoras
        #math.sqrt( (angle_ML*angle_ML) + (angle_AP*angle_AP))
        
        # finally, flip the sign depending on if ML/AP_dist is POSITIVE or NEGATIVE
        if ML_dist < 0:
            angle_ML = -(angle_ML)
        
        if AP_dist < 0:
            angle_AP = -(angle_AP)
        
        # add values to lists
        ml_list.append(angle_ML)
        ap_list.append(angle_AP)
        
    
    # now generate plot - loop through points and plot
    
    # add institution col
    probe_data['inst'] = probe_data['lab'].map(institution_map)
    
    probe_data['ml_angle'] = ml_list
    probe_data['ap_angle'] = ap_list
    probe_data['angle'] = angle_list
    lab_mean_histml = probe_data.groupby('lab')['ml_angle'].mean()
    lab_mean_histap = probe_data.groupby('lab')['ap_angle'].mean()
    
    
    fig1, ax1 = plt.subplots()
    
    # create boxplot with institution colours
    #sns.boxplot(y='inst', x='angle', data=probe_data, 
    #            palette = institution_colors,  orient="h",
    #            ax=ax1)
    
    sns.stripplot(y='inst', x='angle', data=probe_data, 
                  color = 'black', alpha = 0.8, size = 1.5, 
                   orient="h",
                  ax=ax1)
    
    # plot the mean line
    sns.boxplot(showmeans=True,
            meanline=True,
            meanprops={'color': 'gray', 'ls': '-', 'lw': 1},
            medianprops={'visible': False},
            whiskerprops={'visible': False},
            zorder=10,
            x="angle",
            y="inst",
            data=probe_data,
            showfliers=False,
            showbox=False,
            showcaps=False,
            ax=ax1)
    
    #ax1.set_xlabel('Institution', fontsize=7)
    ax1.set_ylabel(None)
    ax1.set_xlim(0, 20)
    ax1.set_xlabel('Histology angle (degrees)')
    
    #ax1.tick_params(axis='x', labelrotation = 90)
    
    ax1.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax1.tick_params(axis='x', labelrotation=90)
    
    plt.tight_layout() # tighten layout around xlabel & ylabel
    
    p_hist1 = permutation_test_histology()
    p_hist2 = permutation_test_histology()
    p_hist3 = permutation_test_histology()
    p_hist = stat.mean([p_hist1, p_hist2, p_hist3])
    
    ax1.set_title('Permutation Test p-value: ' + str( round( p_hist, 4) ) )
    
    fig1.set_size_inches(2.15, 1.8)
    fig1.savefig( str(Path(OUTPUT, 'E_probe_angle_hist_lab.svg')), bbox_inches="tight" )
    #fig1.savefig( str(Path('figure_histology', 'probe-plots','micro_surf_err_plot.svg')), bbox_inches="tight" ) # tight ensures figure is in bounds of svg canvas!
    





def permut_dist(data, labs, mice):
    '''
    Function for computing the permutation test statistic.

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    labs : TYPE
        DESCRIPTION.
    mice : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    import numpy as np
    
    lab_means = []
    for lab in np.unique(labs):
        lab_means.append(np.mean(data[labs == lab]))
    lab_means = np.array(lab_means)
    
    return np.sum(np.abs(lab_means - np.mean(lab_means)))
    



def permutation_test_histology(passed_eids = None):
    '''
    

    Returns
    -------
    None.

    '''
    from pathlib import Path
    import matplotlib.pyplot as plt
    import numpy as np
    import reproducible_ephys_functions as ref
    import figure_hist_data as fhd
    from permutation_test import permut_test
    import math
    from one.api import ONE
    from iblatlas.atlas import Insertion
    import iblatlas.atlas as atlas
    import statistics as stat
    
    # get probe data if necessary
    if 'probe_data' not in vars() or 'probe_data' not in globals():
        # get probe_data histology query
        probe_data = fhd.get_probe_data()
    
    # get histology angle
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
    ins_plan_traj = ins_plan.trajectory
    
    # empty numpy arrays for storing the entry point of probe into brain
     # and "exit point" i.e the probe tip!
    all_ins_entry = np.empty((0, 3))
    
    # get institution map and colours
    lab_number_map, institution_map, institution_colors = ref.labs()
    
    # ins_plan - insertion for planned
    ml_list = list()
    ap_list = list()
    lab_list = list()
    angle_list = list()
    
    # plot line and points from planned insertion entry to actual histology entry
    #for idx in range(len(probe_data)):
    for idx, row in probe_data.iterrows():
        
        phi_lab = row['lab']
        
        lab_list.append(phi_lab)
        
        # get the insertion object for histology
        ins_hist = Insertion.from_dict({ 'x': float(row['hist_x']), 
                             'y': float(row['hist_y']), 
                             'z': float(row['hist_z']), 
                             'phi': float(row['hist_phi']), 
                             'theta': float(row['hist_theta']), 
                             'depth': float(row['hist_depth']) })
        
        # compute the difference in entry coord between ins_plan and ins_hist
        # and use this to move the ins_hist TIP coord
        # ins_plan.xyz[0, :] surface and ins_plan.xyz[1, :] tip
        ins_diff = ins_plan.xyz[0,:] - ins_hist.xyz[0,:] # get difference between surface insertions
        # ins_hist.xyz[0,:] + ins_diff # GIVES ins_plan.xyz[0,:] !!
         # i.e it MOVES the surface coord of ins_hist to ins_plan surface coord!
        hist_tip_norm = ins_hist.xyz[1,:] + ins_diff # gives tip coord, with surface moved to planned ins coord
        # ins_plan.xyz[1,:] ins_hist[1,:] - coords are ML, AP, DV
          # want to IGNORE DV - look at direction between coords in ML, AP
        
        # calculate the ANGLE between ins_plan TIP -> SURFACE -> hist_tip_norm
         # ins_plan.xyz[1, :] -> ins_plan.xyz[0, :] -> hist_tip_norm
        # from: https://stackoverflow.com/questions/35176451/python-code-to-calculate-angle-between-three-point-using-their-3d-coordinates
        ba = ins_plan.xyz[1, :] - ins_plan.xyz[0, :]
        bc = hist_tip_norm - ins_plan.xyz[0, :]
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        
        angle = np.arccos(cosine_angle) # returns angle in RADIANS
        angle_deg = np.degrees(angle) # can use degrees if desired
        
        angle_list.append(angle_deg)
        # want to PLOT the MAGNITUDE of this angle!
        
        # NEXT compute the DIRECTION of the angle from planned to hist
         # get this by first computing the normal vector from hist_tip_norm 
         # to the planned trajectory
    
        # project hist_tip_norm to the PLANNED trajectory:
        hist_tip_norm_vector = ins_plan_traj.project(hist_tip_norm)
        
        # this gives the coord on ins_plan_traj which makes a right angle with
         # hist_tip_norm - so together these coords give the NORMAL VECTOR to
         # ins_plan_traj
        
        # can also  calculate the distance between hist_tip_norm and ins_plan trajectory:
        hist_tip_dist = np.linalg.norm( hist_tip_norm - hist_tip_norm_vector ) *1e6
        
        # from the NORMAL VECTOR can comptue the ML and AP distances by simply subtracting
        # the two values in the vector
        ML_dist = hist_tip_norm[0] - hist_tip_norm_vector[0]
        AP_dist = hist_tip_norm[1] - hist_tip_norm_vector[1]
        
        # and thus can compute the ABSOLUTE RATIO that the ANGLE MAGNITUDE is shared between
        # ML and AP
        ML_ratio = abs(ML_dist) / (abs(ML_dist) + abs(AP_dist))
        AP_ratio = abs(AP_dist) / (abs(ML_dist) + abs(AP_dist))
        
        # combining this ratio with the SIGN of ML_dist/AP_dist can be used to compute the coord
        # of the ANGLE MAGNITUDE from 0,0 as the planned trajectory
        
        # using pythagoras - compute the distance of the hypotenuse 
        # if using ML_ratio and AP_ratio as the right angle lengths
        hyp = math.sqrt( (ML_ratio*ML_ratio) + (AP_ratio*AP_ratio) )
        
        # use this to calculate the proportions of each ratio
        angle_ML = (angle_deg / hyp) * ML_ratio
        angle_AP = (angle_deg / hyp) * AP_ratio
        
        # confirm this works by checking the total length is correct with pythagoras
        #math.sqrt( (angle_ML*angle_ML) + (angle_AP*angle_AP))
        
        # finally, flip the sign depending on if ML/AP_dist is POSITIVE or NEGATIVE
        if ML_dist < 0:
            angle_ML = -(angle_ML)
        
        if AP_dist < 0:
            angle_AP = -(angle_AP)
        
        # add values to lists
        ml_list.append(angle_ML)
        ap_list.append(angle_AP)
        
    
    # now generate plot - loop through points and plot
    
    # add institution col
    probe_data['inst'] = probe_data['lab'].map(institution_map)
    
    probe_data['ml_angle'] = ml_list
    probe_data['ap_angle'] = ap_list
    probe_data['angle'] = angle_list
    
    # filter with passed eids if exists
    if passed_eids is not None:
        probe_data = probe_data[probe_data.eid.isin(passed_eids)]
    
    lab_mean_histml = probe_data.groupby('lab')['ml_angle'].mean()
    lab_mean_histap = probe_data.groupby('lab')['ap_angle'].mean()
    
    p_hist = permut_test( np.array( list(probe_data['angle']) ),
                 metric=permut_dist,
                 labels1 = np.array( list(probe_data['lab']) ),
                 labels2 = np.array( list(probe_data['subject']) ) )
    
    return p_hist
    








def oldfunction():
    
    # ACTUALLY - to frame 
    # direction is computed in ML and AP by looking at the tip coords
    # ins_plan.xyz[1,:] ins_hist[1,:] - coords are ML, AP, DV
    # want to IGNORE DV - look at direction between coords in ML, AP
    
    # so want to calibrate the length between 
    # ins_hist.xyz[1,0:2] and ins_plan.xyz[1,0:2] to ANGLE
    # AND want to ensure ins_plan.xyz[1,0:2] is set to 0,0
    
    all_ins_entry = np.vstack([all_ins_entry, 
                               np.array( ( abs(row['micro_x']/1e6), 
                                          abs(row['micro_y']/1e6), 
                                          abs(row['micro_z']/1e6)) )  ])
    
    ax1.plot( [ abs(row['micro_y']), abs(row['planned_y']) ], 
              [ abs(row['micro_x']), abs(row['planned_x']) ], 
          color= institution_colors[institution_map[phi_lab]], 
          marker="o", markersize=1, linewidth = 0.2 )
        
        # PROJECT the 3 x 3D points to a 2D SURFACE
        # plot this 2D triangle to see the angle from planned insertion to normalised histology tip...
        # this can allow a visualisation of the angle for some subjects/insertions and check data is correct!
        
        # to do this - need to work out lengths and angles of the triangles sides in 3D
        # can then draw this triangle on 2D plane - and can choose the coordinate system in which to draw it
        # LENGTHS
        #AB = 
        #BC = 
        #AC = 
        # ANGLES
        #ABC = 
        #BCA = 
        #CAB = 
        
        # then to plot the triangle in 2D, choose an initial coord in 2D for A
        # then choose the direction from A to B and C
        # computing positions of B and C
        # B is length AB from A ; C is length AC from A; and B is length BC from C
        # CAB is angle at A of vectors AC and AB - this constrains the triangle 
        # can then confirm triangle is correct with ABC and BCA angles
        
    
    
    # plot the planned insertion entry as large blue dot
    #ax1.plot(probe_data['planned_y'][0], probe_data['planned_x'][0], 
    #         color='darkblue', marker="o", markersize=3)
    
    
    # Compute targeting error at surface of brain
    error_top = all_ins_entry - np.array( ( abs(probe_data['planned_x'][0]/1e6), 
                                              abs(probe_data['planned_y'][0]/1e6), 
                                              abs(probe_data['planned_z'][0]/1e6)) )
    distance_top = np.sqrt(np.sum(error_top ** 2, axis=1)) # distance between xyz coords
    top_mean = np.mean(distance_top)*1e6
    top_std = np.std(distance_top)*1e6
    
    rms_top = np.sqrt(np.mean(distance_top ** 2))*1e6
    
    # set x/y axis labels
    ax1.set_xlabel('ap (um)', fontsize=7)
    #ax1.tick_params(axis='x', labelsize=7)
    ax1.set_ylabel('ml (um)', fontsize=7)
    #ax1.tick_params(axis='y', labelsize=7)
    #ax1.yaxis.set_major_locator(plt.MaxNLocator(4))
    
    #ax1.set_ylim((-3000,-1250))
    #ax1.set_xlim((-1000,-3500))
    #ax1.set_ylim((-3000,-500))
    #ax1.set_xlim((-500,-3500))
    ax1.set_ylim((-10,10))
    ax1.set_xlim((-10,10))
    
    # Move left y-axis and bottim x-axis to centre, passing through (0,0)
    ax1.spines['left'].set_position('center')
    ax1.spines['bottom'].set_position('center')
    # Eliminate upper and right axes
    ax1.spines['right'].set_color('none')
    ax1.spines['top'].set_color('none')

    
    plt.tight_layout() # tighten layout around xlabel & ylabel
    
    fig1.set_size_inches(3, 3)
    fig1.savefig( str(Path(OUTPUT, 'D_probe_surf_coord_micro.svg')), bbox_inches="tight" )
    #fig1.savefig( str(Path('figure_histology', 'probe-plots','micro_surf_err_plot.svg')), bbox_inches="tight" ) # tight ensures figure is in bounds of svg canvas!
    
    # add mean trageting error distance to title
    ax1.set_title('MICRO-MANIPULATOR: Mean distance ' +
                  str(np.around(top_mean, 1)) + ' µm', fontsize=10)
    
    fig1.savefig( str(Path(OUTPUT, 'D_probe_surf_coord_micro_label.svg')), bbox_inches="tight" )
    #fig1.savefig( str(Path('figure_histology', 'probe-plots','micromanipulator_surface_error_plot.svg')), bbox_inches="tight" ) # tight ensures figure is in bounds of svg canvas!




def plot_probe_surf_coord_histology(output='figure_histology'):
    '''Plot the PLANNED surface coord at [0,0], VECTORS from planned surface to
    actual surface coord of histology tracks, histology track points coloured
    by lab affiliation.
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
    import reproducible_ephys_functions as ref
    
    # output DIR
    OUTPUT = Path(output)
    
    # generate output DIR for storing plots:
    if os.path.exists(OUTPUT) is False:
        os.mkdir(OUTPUT)
    
    # connect to ibl server
    one = ONE()
    
    # set figure style for consistency
    ref.figure_style()
    
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
    probe_data = probe_geom_data.load_trajectory_data('-2243_-2000')
    
    # get subject IDs as list
    #ids = list(dict.fromkeys(probe_data['subject']))
    
    # manual exclusions
    index = probe_data[ probe_data['subject'] == 'CSH_ZAD_001' ].index
    probe_data.drop(index , inplace=True)
    index = probe_data[ probe_data['subject'] == 'CSHL051' ].index
    probe_data.drop(index , inplace=True)
    index = probe_data[ probe_data['subject'] == 'NYU-47' ].index
    probe_data.drop(index , inplace=True)
    index = probe_data[ probe_data['subject'] == 'UCLA011' ].index
    probe_data.drop(index , inplace=True)
    index = probe_data[ probe_data['subject'] == 'KS051' ].index
    probe_data.drop(index , inplace=True)
    #ids.remove('CSH_ZAD_001') # this has very poor anatomy - dissection damage
    #ids.remove('CSHL051') # this has very poor anatomy - dissection damage
    #ids.remove('NYU-47') # it has no histology ???
    #ids.remove('UCLA011') # currently repeated site WAYYY OUT!
    #ids.remove('KS051') # this sample has not been imaged..
    
    fig1, ax1 = plt.subplots()
    
    
    # empty numpy arrays for storing the entry point of probe into brain
     # and "exit point" i.e the probe tip!
    all_ins_entry = np.empty((0, 3))
    
    # get institution map and colours
    lab_number_map, institution_map, institution_colors = ref.labs()
    
    # track legend insertion in loop
    labs = list(set(probe_data['lab']))
    labs_legend = [False] * len(labs)
    labs_legend[1] = True # set hoferlab to True to prevent two SWCs in the legend
    
    # plot line and points from planned insertion entry to actual histology entry
    #for idx in range(len(probe_data)):
    for idx, row in probe_data.iterrows():
        phi_lab = row['lab']
        all_ins_entry = np.vstack([all_ins_entry, 
                                   np.array( ( abs(row['hist_x']/1e6), 
                                              abs(row['hist_y']/1e6), 
                                              abs(row['hist_z']/1e6)) )  ])
        
        if labs_legend[labs.index(phi_lab)]:
            ax1.plot( [ abs(row['hist_y']), abs(row['planned_y']) ], 
                  [ abs(row['hist_x']), abs(row['planned_x']) ], 
              color= institution_colors[institution_map[phi_lab]], 
              marker="o", markersize=1, linewidth = 0.2)
        else:
            ax1.plot( [ abs(row['hist_y']), abs(row['planned_y']) ], 
                  [ abs(row['hist_x']), abs(row['planned_x']) ], 
              color= institution_colors[institution_map[phi_lab]], 
              marker="o", markersize=1, linewidth = 0.2,
              label = institution_map[phi_lab])
            labs_legend[labs.index(phi_lab)] = True # ensure each lab is added ONCE to legend!
    
    ax1.legend(loc='lower right', prop={'size': 6})
    # plot the planned insertion entry as large blue dot
    #ax1.plot(probe_data['planned_y'][0], probe_data['planned_x'][0], 
    #         color='darkblue', marker="o", markersize=3)
    
    
    # Compute targeting error at surface of brain
    error_top = all_ins_entry - np.array( ( abs(probe_data['planned_x'][0]/1e6), 
                                              abs(probe_data['planned_y'][0]/1e6), 
                                              abs(probe_data['planned_z'][0]/1e6)) )
    distance_top = np.sqrt(np.sum(error_top ** 2, axis=1)) # distance between xyz coords
    top_mean = np.mean(distance_top)*1e6
    top_std = np.std(distance_top)*1e6
    
    rms_top = np.sqrt(np.mean(distance_top ** 2))*1e6
    
    # set x/y axis labels
    ax1.set_xlabel('ap (um)', fontsize=7)
    #ax1.tick_params(axis='x', labelsize=7)
    ax1.set_ylabel('ml (um)', fontsize=7)
    #ax1.tick_params(axis='y', labelsize=7)
    #ax1.yaxis.set_major_locator(plt.MaxNLocator(4))
    
    #ax1.set_ylim((-3000,-1250))
    #ax1.set_xlim((-1000,-3500))
    #ax1.set_ylim((-3000,-500))
    #ax1.set_xlim((-500,-3500))
    ax1.set_ylim((500,5000))
    ax1.set_xlim((1000,4000))

    
    plt.tight_layout() # tighten layout around xlabel & ylabel
    
    fig1.set_size_inches(3, 3)
    fig1.savefig( str(Path(OUTPUT, 'D_probe_surf_coord_hist.svg')), bbox_inches="tight" )
    #fig1.savefig( str(Path('figure_histology', 'probe-plots','micro_surf_err_plot.svg')), bbox_inches="tight" ) # tight ensures figure is in bounds of svg canvas!
    
    # add mean trageting error distance to title
    ax1.set_title('HISTOLOGY: Mean distance ' +
                  str(np.around(top_mean, 1)) + ' µm', fontsize=10)
    
    fig1.savefig( str(Path(OUTPUT, 'D_probe_surf_coord_hist_label.svg')), bbox_inches="tight" )
    #fig1.savefig( str(Path('figure_histology', 'probe-plots','micromanipulator_surface_error_plot.svg')), bbox_inches="tight" ) # tight ensures figure is in bounds of svg canvas!




def old():
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    
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
        #cax.plot(ins.xyz[:, 0] * 1e6, ins.xyz[:, 2] * 1e6, 
        #         color= institution_colors[institution_map[phi_lab]],
        #         linewidth = 0.5)
        #sax.plot(ins.xyz[:, 1] * 1e6, ins.xyz[:, 2] * 1e6, 
        #         color= institution_colors[institution_map[phi_lab]],
        #         linewidth = 0.5)
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

    # add planned insertion ON TOP of the actual insertions, in YELLOW
    cax.plot(ins_plan.xyz[:, 0] * 1e6, ins_plan.xyz[:, 2] * 1e6, 'gold', linewidth=2)
    sax.plot(ins_plan.xyz[:, 1] * 1e6, ins_plan.xyz[:, 2] * 1e6, 'gold', linewidth=2)
    
    ax1.set_ylim((-5000,0))
    ax1.set_xlim((-4000,0))
    ax2.set_ylim((-5000,0))
    ax2.set_xlim((0,-4000))
    
    ax1.tick_params(axis='x', labelrotation = 90)
    ax2.tick_params(axis='x', labelrotation = 90)
    
    #start, end = ax1.get_xlim()
    #ax1.xaxis.set_ticks(np.arange(start, end, 1000)) # ticks every 1000um
    #start, end = ax2.get_xlim()
    #ax2.xaxis.set_ticks(np.arange(start, end, -1000)) # ticks every 1000um
    
    # save to output
    fig1.savefig( str(Path(OUTPUT, 'C_probe_trajs_ccf_coronal.svg')), bbox_inches="tight" )
    fig2.savefig( str(Path(OUTPUT, 'C_probe_trajs_ccf_sagittal.svg')), bbox_inches="tight" )
    


def plot_trajs_cor_sag_hor(output='figure_histology'):
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



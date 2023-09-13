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
    



def plot_probe_surf_coord_micro_panel(output = 'figure_histology'):
    '''
    Plot the whole probe micro-manipulator panel, consisting of: 
    
    * scatterplot of the PLANNED to MICRO coords at brain surface, 
    
    * horizontal boxplot plus distplot (density plot) of all PLANNED to 
    MICRO surf coord distances (to see total distribution),
    
    * horizontal boxplots of each labs distribution
    
    * heat map of each labs permutation test p-value.?
    
    
    Panel saved to output as: surf_coord_micro_panel.svg

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
    plot_probe_surf_coord_micro(probe_data, output=output) # saves as SVG to output
    
    # generate histogram/density plot of Euclidean distance at surface from 
    # planned to actual for all trajectories AND dotplots by lab
    plot_probe_distance_micro_all_lab(probe_data, output=output)
    
    # generate grouped boxplots of Euclidean distance at surface from 
    # planned to actual for all trajectories, by LAB
    #plot_probe_distance_micro_lab(probe_data, output=output)
    
    
    # generate horizontal boxplot plus density plot of all surf coord histology
     # in second axes
    #plot_probe_distance_histology_all(probe_data, ax2)
    fig = sc.Figure( "66mm", "140mm",
        
        sc.Panel(
            sc.SVG(output+os.path.sep+'D_probe_surf_coord_micro_label.svg'
                   ).scale(0.35)
            ),
        
        sc.Panel(
            sc.SVG(output+os.path.sep+'D_probe_dist_micro_all_lab.svg'
                ).scale(0.35
                ).move(0, 68)
            ),
        
        #sc.Grid(20, 20)
        )
    
    fig.save( Path(output, "surf_coord_micro_panel.svg") )
    



def plot_probe_surf_coord_micro(probe_data, output='figure_histology'):
    '''Plot the PLANNED surface coord at [0,0], VECTORS from planned surface to
    actual surface coord of histology tracks, histology track points coloured
    by lab affiliation.
    '''
    from pathlib import Path
    import os
    import figure_hist_data as fhd
    import matplotlib.pyplot as plt
    #from one.api import ONE
    #from iblatlas.atlas import Insertion
    import numpy as np
    #import atlaselectrophysiology.load_histology as hist
    #import iblatlas.atlas as atlas
    import reproducible_ephys_functions as ref
    
    # use repo-ephys figure style
    ref.figure_style()
    
    # get probe data if necessary
    if 'probe_data' not in vars() or 'probe_data' not in globals():
        # get probe_data histology query
        probe_data = fhd.get_probe_data()
    
    # output DIR - generate output DIR for storing plots:
    OUTPUT = Path(output)
    if os.path.exists(OUTPUT) is False:
        os.mkdir(OUTPUT)
    
    # create new column to indicate if each row passes advanced query
    passed_eids = ref.eid_list()
    # permutation testing - PASS DATA ONLY
    probe_data_passed = probe_data[probe_data.eid.isin(passed_eids)]
    
    # get all trajectories at the REPEATED SITE
     # for ibl brainwide map project
     # repeated site location: x -2243 y -2000
    #traj = one.alyx.rest('trajectories', 'list', provenance='Planned',
    #                     x=-2243, y=-2000,  project='ibl_neuropixel_brainwide_01')
    
    # get eids, probe names and subject names from traj
    #eids = [sess['session']['id'] for sess in traj]
    #probes = [sess['probe_name'] for sess in traj]
    
    # Get the trajectory for the planned repeated site recording
    #phi_eid = eids[0]
    #phi_probe = probes[0]
    #phi_traj = one.alyx.rest('trajectories', 'list', session=phi_eid,
    #                         provenance='Planned', probe=phi_probe)[0]
    # planned as insertion: includes XYZ and phi,theta,depth data
    #ins_plan = atlas.Insertion.from_dict(phi_traj)
    
    # get new atlas for plotting
    #brain_atlas = atlas.AllenAtlas(res_um=25)
    
    # main panel figure
    fig1, ax1 = plt.subplots()
    
    # draw 0,0 lines
    ax1.axhline(y=-2000, color="grey", linestyle="--", linewidth = 0.5)
    ax1.axvline(x=-2243, color="grey", linestyle="--", linewidth = 0.5)
    
    # empty numpy arrays for storing the entry point of probe into brain
     # and "exit point" i.e the probe tip!
    all_ins_entry = np.empty((0, 3))
    
    # FIRST just get all ins entry for PASSED probes to compute mean(SD) distance
    all_ins_entry_pass = np.empty((0, 3))
    
    for idx, row in probe_data_passed.iterrows():
        phi_lab = row['lab']
        all_ins_entry_pass = np.vstack([all_ins_entry_pass, 
                                   np.array( ( abs(row['micro_x']/1e6), 
                                              abs(row['micro_y']/1e6), 
                                              abs(row['micro_z']/1e6)) )  ])
    
    # get institution map and colours
    lab_number_map, institution_map, institution_colors = ref.labs()
    
    # plot line and points from planned insertion entry to actual histology entry
    #for idx in range(len(probe_data)):
    for idx, row in probe_data.iterrows():
        
        phi_lab = row['lab']
        all_ins_entry = np.vstack([all_ins_entry, 
                                   np.array( ( abs(row['micro_x']/1e6), 
                                              abs(row['micro_y']/1e6), 
                                              abs(row['micro_z']/1e6)) )  ])
        
        # plot x(ML) and y (AP) on x and y axes
        ax1.plot( [ row['micro_x'], row['planned_x'] ], 
                  [ row['micro_y'], row['planned_y'] ], 
              color= institution_colors[institution_map[phi_lab]], 
              linewidth = 0.15, alpha = 0.8 )
        
        ax1.plot( [ row['micro_x']], 
                  [ row['micro_y'] ], 
              color= institution_colors[institution_map[phi_lab]], 
              marker="o", markersize=0.5, alpha = 0.8,  markeredgewidth = 0.5)
    
    
    # plot the planned insertion entry as large blue dot
    #ax1.plot(probe_data['planned_y'][0], probe_data['planned_x'][0], 
    #         color='darkblue', marker="o", markersize=3)
    
    # plot the mean micro coords
    
    # lab means
    lab_mean_microx = probe_data.groupby('lab')['micro_x'].mean()
    lab_mean_microy = probe_data.groupby('lab')['micro_y'].mean()
    
    for x, y, k in zip(lab_mean_microx, lab_mean_microy, lab_mean_microx.keys()):
        ax1.plot( [ x ], 
                  [ y ], 
              color= institution_colors[institution_map[k]], 
              marker="+", markersize=3, alpha = 0.5,
              label = institution_map[k])
    
    # overall mean (mean of labs)
    mean_microx = probe_data['micro_x'].mean()
    mean_microy = probe_data['micro_y'].mean()
    
    ax1.plot( [ mean_microx ], 
                  [ mean_microy ], 
              color= 'k', marker="+", markersize=6, alpha = 0.7,
              label = "MEAN")
    
    # add legend
    ax1.legend(loc='upper right', prop={'size': 3.5})
    
    # Compute targeting error at surface of brain
    error_top = all_ins_entry - np.array( ( abs(probe_data['planned_x'][0]/1e6), 
                                              abs(probe_data['planned_y'][0]/1e6), 
                                              abs(probe_data['planned_z'][0]/1e6)) )
    distance_top = np.sqrt(np.sum(error_top ** 2, axis=1)) # distance between xyz coords
    top_mean = np.mean(distance_top)*1e6
    top_std = np.std(distance_top)*1e6
    
    rms_top = np.sqrt(np.mean(distance_top ** 2))*1e6
    
    # error for PASSING probes only
    error_top_pass = all_ins_entry_pass - np.array( ( abs(probe_data_passed['planned_x'][0]/1e6), 
                                              abs(probe_data_passed['planned_y'][0]/1e6), 
                                              abs(probe_data_passed['planned_z'][0]/1e6)) )
    distance_top_pass = np.sqrt(np.sum(error_top_pass ** 2, axis=1)) # distance between xyz coords
    top_mean_pass = np.mean(distance_top_pass)*1e6
    top_std_pass = np.std(distance_top_pass)*1e6
    
    rms_top_pass = np.sqrt(np.mean(distance_top_pass ** 2))*1e6
    
    # set x/y axis labels
    ax1.set_xlabel('micro-manipulator ML displacement (µm)', fontsize=6)
    #ax1.tick_params(axis='x', labelsize=7)
    ax1.set_ylabel('micro-manipulator AP displacement (µm)', fontsize=6)
    #ax1.tick_params(axis='y', labelsize=7)
    #ax1.yaxis.set_major_locator(plt.MaxNLocator(4))
    
    #ax1.set_ylim((1800,2600))
    #ax1.set_xlim((1600,2300))
    ax1.set_xlim((-2800,-800))
    ax1.set_ylim((-3000,-1000))
    ax1.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax1.yaxis.set_major_locator(plt.MaxNLocator(5))
    
    plt.tight_layout() # tighten layout around xlabel & ylabel
    
    fig1.set_size_inches(2.15, 2.15)
    
    # add a subplot INSIDE the fig1 ax1
    axav = fig1.add_axes([0.66,0.12,0.28,0.28])
    
    #axav.axes.xaxis.set_ticks([])
    #axav.axes.yaxis.set_ticks([])
    axav.xaxis.tick_top()
    axav.tick_params(axis='both', labelsize=3, pad = 1)
    
    axav.axhline(y=-2000, color="grey", linestyle="--", linewidth = 0.5)
    axav.axvline(x=-2243, color="grey", linestyle="--", linewidth = 0.5)
    axav.set_xlim((-2350,-2000))
    axav.set_ylim((-2100,-1850))
    
    for x, y, k in zip(lab_mean_microx, lab_mean_microy, lab_mean_microx.keys()):
        axav.plot( [ x ], 
                  [ y ], 
              color= institution_colors[institution_map[k]], 
              marker="+", markersize=5, alpha = 0.7,
              label = institution_map[k])
    
    axav.plot( [ mean_microx ], 
                  [ mean_microy ], 
              color= 'k', marker="+", markersize=8, alpha = 0.7,
              label = "MEAN")
    
    #axav.tight_layout()
    
    fig1.savefig( str(Path(OUTPUT, 'D_probe_surf_coord_micro.svg')), bbox_inches="tight" )
    #fig1.savefig( str(Path('figure_histology', 'probe-plots','micro_surf_err_plot.svg')), bbox_inches="tight" ) # tight ensures figure is in bounds of svg canvas!
    
    # add mean trageting error distance to title
    ax1.set_title('MICRO-MANIPULATOR: Mean (SD) distance \n' +
                  str(np.around(top_mean, 1)) + ' ('+str(np.around(top_std, 2))+')'+ ' µm\n' +
                  'PASSED: ' +str(np.around(top_mean_pass, 0)) + 
                  ' ('+str(np.around(top_std_pass, 0))+')' + ' µm', fontsize=8)
    
    fig1.savefig( str(Path(OUTPUT, 'D_probe_surf_coord_micro_label.svg')), bbox_inches="tight" )
    #fig1.savefig( str(Path('figure_histology', 'probe-plots','micromanipulator_surface_error_plot.svg')), bbox_inches="tight" ) # tight ensures figure is in bounds of svg canvas!



def plot_probe_distance_micro_all_lab(probe_data, output='figure_histology'):
    '''Plot the DISTANCES from planned to micro displacement, histogram plus
    density plot of ALL distances - to see its distribution shape.
    COMBINED with plot of distances, split by lab
    '''
    from pathlib import Path
    import os
    #from probe_geometry_analysis import probe_geometry_data as probe_geom_data
    import matplotlib.pyplot as plt
    import seaborn as sns
    #import numpy as np
    import reproducible_ephys_functions as ref
    import figure_hist_data as fhd
    import math
    import statistics as stat
    
    ref.figure_style()
    
    # get probe data if necessary
    if 'probe_data' not in vars() or 'probe_data' not in globals():
        # get probe_data histology query
        probe_data = fhd.get_probe_data()
    
    # output DIR - generate output DIR for storing plots:
    OUTPUT = Path(output)
    if os.path.exists(OUTPUT) is False:
        os.mkdir(OUTPUT)
    
    # get institution map and colours
    lab_number_map, institution_map, institution_colors = ref.labs()
    
    # add institution col
    probe_data['inst'] = probe_data['lab'].map(institution_map)
    
    # create boxplot with institution colours
    #sns.boxplot(x='hist_error_surf', data=probe_data, 
    #            color = 'white',  orient="h",
    #            ax=ax1)
    #sns.stripplot(x='hist_error_surf', data=probe_data, 
    #              color = 'black', alpha = 0.8, size = 3, 
    #               orient="h",
    #              ax=ax1)
    # density plot
    
    # compute the histology distance
    x_dist = abs(abs(probe_data['micro_x']) - abs(probe_data['planned_x']))**2
    y_dist = abs(abs(probe_data['micro_y']) - abs(probe_data['planned_y']))**2
    sum_dist = x_dist + y_dist
    dist_list = []
    for s in sum_dist:
        dist_list.append(math.sqrt( s ) )
    
    probe_data['micro_dist'] = dist_list
    
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
    colors = ["#000000", "#FF0B04"] # BLACK AND RED
    # Set your custom color palette
    sns.set_palette(sns.color_palette(colors))
    
    fig, (ax1, ax2) = plt.subplots(2, gridspec_kw={'height_ratios': [1, 2]})
    
    sns.histplot(probe_data['micro_dist'], kde=True, 
                 color = 'grey',
                 ax = ax1)
    
    #sns.kdeplot(x = 'micro_dist', data = probe_data, 
    #             hue = 'passed_adv', 
    #             ax = ax2)
    
    
    ax1.set_xlim(0, 1500)
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
    
    sns.stripplot(y='inst', x='micro_dist', data=probe_data, 
                  hue = 'passed_adv', 
                  size = 1.5, 
                  orient="h", 
                  ax=ax2)
    
    # plot the mean line
    sns.boxplot(showmeans=True,
            meanline=True,
            meanprops={'color': 'gray', 'ls': '-', 'lw': 1},
            medianprops={'visible': False},
            whiskerprops={'visible': False},
            zorder=10,
            x="micro_dist",
            y="inst",
            data=probe_data,
            showfliers=False,
            showbox=False,
            showcaps=False,
            ax=ax2)
    
    #ax1.set_xlabel('Institution', fontsize=7)
    ax2.set_ylabel(None)
    ax2.set_xlim(0, 1500)
    ax2.set_xlabel('Micromanipulator distance (µm)')
    #ax2.get_legend().remove()
    l = ax2.legend(fontsize=4, title='Advanced \n query', 
               title_fontsize=6, 
               loc='upper right',
               markerscale = 0.2 )
    plt.setp(l.get_title(), multialignment='center')
    #leg.set_title('Passed adv. query',prop={'size':7})
    
    #ax1.tick_params(axis='x', labelrotation = 90)
    
    ax2.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax2.tick_params(axis='x', labelrotation=90)
    
    # compute permutation testing - ALL DATA
    p_m1 = permutation_test_micro(probe_data)
    p_m2 = permutation_test_micro(probe_data)
    p_m3 = permutation_test_micro(probe_data)
    p_m = stat.mean([p_m1, p_m2, p_m3])
    
    print("PERMUTATION TEST ALL : ", p_m)
    # permutation testing - PASS DATA ONLY
    probe_data_passed = probe_data[probe_data.eid.isin(passed_eids)]
    pp_m1 = permutation_test_micro(probe_data_passed)
    pp_m2 = permutation_test_micro(probe_data_passed)
    pp_m3 = permutation_test_micro(probe_data_passed)
    pp_m = stat.mean([pp_m1, pp_m2, pp_m3])
    print("PERMUTATION TEST PASS : ", pp_m)
    
    ax1.set_title('Permutation Test p-value: \n    ALL : ' 
                    + str( round( p_m, 4) )
                    + '    PASS : ' + str( round( pp_m, 4) ) )
    
    plt.tight_layout() # tighten layout around xlabel & ylabel
    
    fig.set_size_inches(2.15, 2.8)
    fig.savefig( str(Path(output, 'D_probe_dist_micro_all_lab.svg')), bbox_inches="tight" )
    #fig1.savefig( str(Path('figure_histology', 'probe-plots','micro_surf_err_plot.svg')), bbox_inches="tight" ) # tight ensures figure is in bounds of svg canvas!



def plot_probe_distance_micro_all(probe_data, output='figure_histology'):
    '''OLD - Plot the DISTANCES from planned to micro displacement, histogram plus
    density plot of ALL distances - to see its distribution shape.
    '''
    from pathlib import Path
    import os
    #from probe_geometry_analysis import probe_geometry_data as probe_geom_data
    import matplotlib.pyplot as plt
    import seaborn as sns
    #import numpy as np
    import reproducible_ephys_functions as ref
    import figure_hist_data as fhd
    import math
    
    ref.figure_style()
    
    # get probe data if necessary
    if 'probe_data' not in vars() or 'probe_data' not in globals():
        # get probe_data histology query
        probe_data = fhd.get_probe_data()
    
    # output DIR - generate output DIR for storing plots:
    OUTPUT = Path(output)
    if os.path.exists(OUTPUT) is False:
        os.mkdir(OUTPUT)
    
    # get institution map and colours
    lab_number_map, institution_map, institution_colors = ref.labs()
    
    # add institution col
    probe_data['inst'] = probe_data['lab'].map(institution_map)
    
    # create boxplot with institution colours
    #sns.boxplot(x='hist_error_surf', data=probe_data, 
    #            color = 'white',  orient="h",
    #            ax=ax1)
    #sns.stripplot(x='hist_error_surf', data=probe_data, 
    #              color = 'black', alpha = 0.8, size = 3, 
    #               orient="h",
    #              ax=ax1)
    # density plot
    x_dist = abs(abs(probe_data['micro_x']) - abs(probe_data['planned_x']))**2
    y_dist = abs(abs(probe_data['micro_y']) - abs(probe_data['planned_y']))**2
    sum_dist = x_dist + y_dist
    dist_list = []
    for s in sum_dist:
        dist_list.append(math.sqrt( s ) )
    
    probe_data['micro_dist'] = dist_list
    
    fig = sns.displot(probe_data['micro_dist'], kde=True)
    fig1 = fig.fig # fig is a FacetGrid object!
    ax1 = fig.ax # get axes
    ax1.set_xlim(0, 1500)
    #ax1.set_xlabel('Institution', fontsize=7)
    ax1.set_ylabel(None)
    #ax1.set_xlabel('Micro-Manipulator distance (µm)')
    ax1.set_xlabel(None)
    
    #ax1.tick_params(axis='x', labelrotation = 90)
    
    ax1.xaxis.set_major_locator(plt.MaxNLocator(5))
    #ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90, ha='right')
    ax1.tick_params(axis='x', labelrotation=90)
    #ax1.set(yticklabels=[])
    #ax1.tick_params(left=False)
    #ax1.set_axis_off()
    
    plt.tight_layout() # tighten layout around xlabel & ylabel
    
    fig1.set_size_inches(2.15, 1)
    fig1.savefig( str(Path(output, 'D_probe_dist_micro_all.svg')), bbox_inches="tight" )
    #fig1.savefig( str(Path('figure_histology', 'probe-plots','micro_surf_err_plot.svg')), bbox_inches="tight" ) # tight ensures figure is in bounds of svg canvas!



def plot_probe_distance_micro_lab(probe_data, output='figure_histology'):
    '''OLD - Plot the DISTANCES from planned to micro, boxplots coloured
    by lab affiliation.  Add inter-lab means group to plot too?
    '''
    from pathlib import Path
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import reproducible_ephys_functions as ref
    import figure_hist_data as fhd
    import math
    import statistics as stat
    
    # output DIR - generate output DIR for storing plots:
    OUTPUT = Path(output)
    if os.path.exists(OUTPUT) is False:
        os.mkdir(OUTPUT)
    
    # set figure style for consistency
    ref.figure_style()
    
    # get probe data if necessary
    if 'probe_data' not in vars() or 'probe_data' not in globals():
        # get probe_data histology query
        probe_data = fhd.get_probe_data()
    
    # get fig and axes
    fig1, ax1 = plt.subplots()
    
    # get institution map and colours
    lab_number_map, institution_map, institution_colors = ref.labs()
    
    # add institution col
    probe_data['inst'] = probe_data['lab'].map(institution_map)
    
    x_dist = abs(abs(probe_data['micro_x']) - abs(probe_data['planned_x']))**2
    y_dist = abs(abs(probe_data['micro_y']) - abs(probe_data['planned_y']))**2
    sum_dist = x_dist + y_dist
    dist_list = []
    for s in sum_dist:
        dist_list.append(math.sqrt( s ) )
    
    probe_data['micro_dist'] = dist_list
    
    # create boxplot with institution colours
    #sns.boxplot(y='inst', x='micro_dist', data=probe_data, 
    #            palette = institution_colors,  orient="h",
    #            ax=ax1)
    
    sns.stripplot(y='inst', x='micro_dist', data=probe_data, 
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
            x="micro_dist",
            y="inst",
            data=probe_data,
            showfliers=False,
            showbox=False,
            showcaps=False,
            ax=ax1)
    
    #ax1.set_xlabel('Institution', fontsize=7)
    ax1.set_ylabel(None)
    ax1.set_xlim(0, 1500)
    ax1.set_xlabel('Micro-Manipulator distance (µm)')
    
    #ax1.tick_params(axis='x', labelrotation = 90)
    
    ax1.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax1.tick_params(axis='x', labelrotation=90)
    
    plt.tight_layout() # tighten layout around xlabel & ylabel
    
    p_micro1 = permutation_test_micro()
    p_micro2 = permutation_test_micro()
    p_micro3 = permutation_test_micro()
    p_micro = stat.mean([p_micro1, p_micro2, p_micro3])
    
    ax1.set_title('Permutation Test p-value: ' + str( round(p_micro, 4) ) )
    
    fig1.set_size_inches(2.15, 1.8)
    fig1.savefig( str(Path(OUTPUT, 'D_probe_dist_micro_lab.svg')), bbox_inches="tight" )
    #fig1.savefig( str(Path('figure_histology', 'probe-plots','micro_surf_err_plot.svg')), bbox_inches="tight" ) # tight ensures figure is in bounds of svg canvas!
    




def plot_probe_surf_coord_histology_panel(output = 'figure_histology'):
    '''
    Plot the whole probe histology panel, consisting of: 
    
    * scatterplot of the PLANNED to HISTOLOGY coords at brain surface, 
    
    * horizontal boxplot plus distplot (density plot) of all PLANNED to 
    HISTOLOGY surf coord distances (to see total distribution),
    
    * horizontal boxplots of each labs distribution
    
    * heat map of each labs permutation test p-value.?
    
    
    Panel saved to output as: surf_coord_histology_panel.svg

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
    plot_probe_surf_coord_histology(probe_data, output=output) # saves as SVG to output
    
    # generate histogram/density plot of Euclidean distance at surface from 
    # planned to actual for all trajectories AND dotplots by lab
    plot_probe_distance_histology_all_lab(probe_data, output=output)
    
    # generate grouped boxplots of Euclidean distance at surface from 
    # planned to actual for all trajectories, by LAB
    #plot_probe_distance_histology_lab(probe_data, output=output)
    
    
    # generate horizontal boxplot plus density plot of all surf coord histology
     # in second axes
    #plot_probe_distance_histology_all(probe_data, ax2)
    fig = sc.Figure( "66mm", "140mm",
        
        sc.Panel(
            sc.SVG(output+os.path.sep+'D_probe_surf_coord_hist_label.svg'
                   ).scale(0.35)
            ),
        
        sc.Panel(
            sc.SVG(output+os.path.sep+'D_probe_dist_hist_all_lab.svg'
                ).scale(0.35
                ).move(0,68)
            ),
        
        #sc.Grid(20, 20)
        )
    
    fig.save( Path(output, "surf_coord_histology_panel.svg") )
    




def plot_probe_surf_coord_histology(probe_data, output='figure_histology'):
    '''Plot the PLANNED surface coord at [0,0], VECTORS from planned surface to
    actual surface coord of histology tracks, histology track points coloured
    by lab affiliation.
    '''
    from pathlib import Path
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    import reproducible_ephys_functions as ref
    import figure_hist_data as fhd
    
    ref.figure_style()
    
    # get probe data if necessary
    if 'probe_data' not in vars() or 'probe_data' not in globals():
        # get probe_data histology query
        probe_data = fhd.get_probe_data()
    
    # output DIR - generate output DIR for storing plots:
    OUTPUT = Path(output)
    if os.path.exists(OUTPUT) is False:
        os.mkdir(OUTPUT)
    
    # create new column to indicate if each row passes advanced query
    passed_eids = ref.eid_list()
    # permutation testing - PASS DATA ONLY
    probe_data_passed = probe_data[probe_data.eid.isin(passed_eids)]
    
    # generate figure and axes
    fig, ax = plt.subplots()
    
    # draw 0,0 lines
    ax.axhline(y=-2000, color="grey", linestyle="--", linewidth = 0.5)
    ax.axvline(x=-2243, color="grey", linestyle="--", linewidth = 0.5)
    
    # empty numpy arrays for storing the entry point of probe into brain
     # and "exit point" i.e the probe tip!
    all_ins_entry = np.empty((0, 3))
    
    # FIRST just get all ins entry for PASSED probes to compute mean(SD) distance
    all_ins_entry_pass = np.empty((0, 3))
    
    for idx, row in probe_data_passed.iterrows():
        phi_lab = row['lab']
        all_ins_entry_pass = np.vstack([all_ins_entry_pass, 
                                   np.array( ( abs(row['hist_x']/1e6), 
                                              abs(row['hist_y']/1e6), 
                                              abs(row['hist_z']/1e6)) )  ])
    
    # get institution map and colours
    lab_number_map, institution_map, institution_colors = ref.labs()
    
    # track legend insertion in loop
    #labs = list(set(probe_data['lab']))
    #labs_legend = [False] * len(labs)
    #labs_legend[1] = True # set hoferlab to True to prevent two SWCs in the legend
    
    # normalise data
    #probe_data['planned_x'] = probe_data['planned_x'] - probe_data['planned_x']
    #probe_data['hist_x'] = probe_data['hist_x'] - probe_data['planned_x']
    #probe_data['planned_y'] = probe_data['planned_y'] - probe_data['planned_y']
    #probe_data['hist_y'] = probe_data['hist_y'] - probe_data['planned_y']
    
    # plot line and points from planned insertion entry to actual histology entry
    #for idx in range(len(probe_data)):
    for idx, row in probe_data.iterrows():
        phi_lab = row['lab']
        all_ins_entry = np.vstack([all_ins_entry, 
                                   np.array( ( abs(row['hist_x']/1e6), 
                                              abs(row['hist_y']/1e6), 
                                              abs(row['hist_z']/1e6)) )  ])
        
        ax.plot( [ row['hist_x'], row['planned_x'] ], 
                  [ row['hist_y'], row['planned_y'] ], 
              color= institution_colors[institution_map[phi_lab]], 
              linewidth = 0.15, alpha = 0.8)
        ax.plot( [ row['hist_x'] ], 
              [ row['hist_y'] ], 
          color= institution_colors[institution_map[phi_lab]], 
          marker="o", markersize=0.5, alpha = 0.8, markeredgewidth = 0.5)
        
        # removing legend per subject - adding it with MEAN PLOTS
        #if labs_legend[labs.index(phi_lab)]:
        #    ax1.plot( [ abs(row['hist_y']), abs(row['planned_y']) ], 
        #          [ abs(row['hist_x']), abs(row['planned_x']) ], 
        #      color= institution_colors[institution_map[phi_lab]], 
        #      linewidth = 0.2)
        #    ax1.plot( [ abs(row['hist_y']) ], 
        #          [ abs(row['hist_x']) ], 
        #      color= institution_colors[institution_map[phi_lab]], 
        #      marker="o", markersize=0.5)
        #else:
        #    ax1.plot( [ abs(row['hist_y']), abs(row['planned_y']) ], 
        #          [ abs(row['hist_x']), abs(row['planned_x']) ], 
        #      color= institution_colors[institution_map[phi_lab]], 
        #      linewidth = 0.2,
        #      label = institution_map[phi_lab])
        #    ax1.plot( [ abs(row['hist_y']) ], 
        #          [ abs(row['hist_x']) ], 
        #      color= institution_colors[institution_map[phi_lab]], 
        #      marker="o", markersize=0.5)
        #    labs_legend[labs.index(phi_lab)] = True # ensure each lab is added ONCE to legend!
    
    # plot the planned insertion entry as large blue dot
    #ax1.plot(probe_data['planned_y'][0], probe_data['planned_x'][0], 
    #         color='darkblue', marker="o", markersize=3)
    
    # plot the mean histology coords
    
    # lab means
    lab_mean_histx = probe_data.groupby('lab')['hist_x'].mean()
    lab_mean_histy = probe_data.groupby('lab')['hist_y'].mean()
    
    for x, y, k in zip(lab_mean_histx, lab_mean_histy, lab_mean_histx.keys()):
        ax.plot( [ x ], 
                  [ y ], 
              color= institution_colors[institution_map[k]], 
              marker="+", markersize=3, alpha = 0.5,
              label = institution_map[k])
    
    # overall mean (mean of labs)
    mean_histx = probe_data['hist_x'].mean()
    mean_histy = probe_data['hist_y'].mean()
    
    ax.plot( [ mean_histx ], 
                  [ mean_histy ], 
              color= 'k', marker="+", markersize=6, alpha = 0.7,
              label = "MEAN")
    
    
    # Compute targeting error at surface of brain
    error_top = all_ins_entry - np.array( ( abs(probe_data['planned_x'][0]/1e6), 
                                              abs(probe_data['planned_y'][0]/1e6), 
                                              abs(probe_data['planned_z'][0]/1e6)) )
    distance_top = np.sqrt(np.sum(error_top ** 2, axis=1)) # distance between xyz coords
    top_mean = np.mean(distance_top)*1e6
    top_std = np.std(distance_top)*1e6
    
    # error for PASSING probes only
    error_top_pass = all_ins_entry_pass - np.array( ( abs(probe_data_passed['planned_x'][0]/1e6), 
                                              abs(probe_data_passed['planned_y'][0]/1e6), 
                                              abs(probe_data_passed['planned_z'][0]/1e6)) )
    distance_top_pass = np.sqrt(np.sum(error_top_pass ** 2, axis=1)) # distance between xyz coords
    top_mean_pass = np.mean(distance_top_pass)*1e6
    top_std_pass = np.std(distance_top_pass)*1e6
    
    rms_top_pass = np.sqrt(np.mean(distance_top_pass ** 2))*1e6
    
    # set x/y axis labels
    ax.set_xlabel('histology ML displacement (µm)', fontsize=6)
    #ax1.tick_params(axis='x', labelsize=7)
    ax.set_ylabel('histology AP displacement (µm)', fontsize=6)
    #ax1.tick_params(axis='y', labelsize=7)
    #ax1.yaxis.set_major_locator(plt.MaxNLocator(4))
    
    #ax1.set_ylim((-3000,-1250))
    #ax1.set_xlim((-1000,-3500))
    #ax1.set_ylim((-3000,-500))
    #ax1.set_xlim((-500,-3500))
    ax.set_xlim((-2800,-800))
    ax.set_ylim((-3000,-1000))
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    
    plt.tight_layout() # tighten layout around xlabel & ylabel
    
    fig.set_size_inches(2.15, 2.15)
    
    # add a subplot INSIDE the fig1 ax1
    axav = fig.add_axes([0.66,0.12,0.28,0.28])
    
    #axav.axes.xaxis.set_ticks([])
    #axav.axes.yaxis.set_ticks([])
    axav.xaxis.tick_top()
    axav.tick_params(axis='both', labelsize=3, pad = 1)
    #axav.xaxis.labelpad = 1
    #axav.yaxis.labelpad = 1
    
    axav.axhline(y=-2000, color="grey", linestyle="--", linewidth = 0.5)
    axav.axvline(x=-2243, color="grey", linestyle="--", linewidth = 0.5)
    axav.set_xlim((-2500,-1650))
    axav.set_ylim((-2400,-1550))
    
    for x, y, k in zip(lab_mean_histx, lab_mean_histy, lab_mean_histx.keys()):
        axav.plot( [ x ], 
                  [ y ], 
              color= institution_colors[institution_map[k]], 
              marker="+", markersize=5, alpha = 0.7,
              label = institution_map[k])
    
    axav.plot( [ mean_histx ], 
                  [ mean_histy ],
              color= 'k', marker="+", markersize=8, alpha = 0.7,
              label = "MEAN")
    
    fig.savefig( str(Path(output, 'D_probe_surf_coord_hist.svg')), bbox_inches="tight" )
    #fig1.savefig( str(Path('figure_histology', 'probe-plots','micro_surf_err_plot.svg')), bbox_inches="tight" ) # tight ensures figure is in bounds of svg canvas!
    
    # add mean trageting error distance to title
    ax.set_title('HISTOLOGY: Mean (SD) distance \n' +
                  'ALL: ' +str(np.around(top_mean, 0)) + 
                  ' ('+str(np.around(top_std, 0))+')'+ ' µm\n' +
                  'PASSED: ' +str(np.around(top_mean_pass, 0)) + 
                  ' ('+str(np.around(top_std_pass, 0))+')' + ' µm', fontsize=8)
    
    fig.savefig( str(Path(OUTPUT, 'D_probe_surf_coord_hist_label.svg')), bbox_inches="tight" )
    #fig1.savefig( str(Path('figure_histology', 'probe-plots','micromanipulator_surface_error_plot.svg')), bbox_inches="tight" ) # tight ensures figure is in bounds of svg canvas!



def plot_probe_distance_histology_all_lab(probe_data, output='figure_histology'):
    '''Plot the DISTANCES from planned to histology displacement, histology 
    track boxplott of ALL angles - to see its distribution shape.
    COMBINED with plot of distances split by LAB.
    '''
    from pathlib import Path
    import os
    #from probe_geometry_analysis import probe_geometry_data as probe_geom_data
    import matplotlib.pyplot as plt
    import seaborn as sns
    #import numpy as np
    import reproducible_ephys_functions as ref
    import figure_hist_data as fhd
    import math
    import statistics as stat
    
    ref.figure_style()
    
    # get probe data if necessary
    if 'probe_data' not in vars() or 'probe_data' not in globals():
        # get probe_data histology query
        probe_data = fhd.get_probe_data()
    
    # output DIR - generate output DIR for storing plots:
    OUTPUT = Path(output)
    if os.path.exists(OUTPUT) is False:
        os.mkdir(OUTPUT)
    
    # get institution map and colours
    lab_number_map, institution_map, institution_colors = ref.labs()
    
    # add institution col
    probe_data['inst'] = probe_data['lab'].map(institution_map)
    
    # create boxplot with institution colours
    #sns.boxplot(x='hist_error_surf', data=probe_data, 
    #            color = 'white',  orient="h",
    #            ax=ax1)
    #sns.stripplot(x='hist_error_surf', data=probe_data, 
    #              color = 'black', alpha = 0.8, size = 3, 
    #               orient="h",
    #              ax=ax1)
    # density plot
    
    # compute the histology distance
    x_dist = abs(abs(probe_data['hist_x']) - abs(probe_data['planned_x']))**2
    y_dist = abs(abs(probe_data['hist_y']) - abs(probe_data['planned_y']))**2
    sum_dist = x_dist + y_dist
    dist_list = []
    for s in sum_dist:
        dist_list.append(math.sqrt( s ) )
    
    probe_data['hist_dist'] = dist_list
    
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
    
    sns.histplot(probe_data['hist_dist'], kde=True, 
                 color = 'grey',
                 ax = ax1)
    
    #sns.kdeplot(x = 'hist_dist', data = probe_data, 
    #             hue = 'passed_adv', 
    #             ax = ax2)
    
    
    ax1.set_xlim(0, 1500)
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
    
    sns.stripplot(y='inst', x='hist_dist', data=probe_data, 
                  hue = 'passed_adv', 
                  size = 1.5, 
                  orient="h", 
                  ax=ax2)
    
    # plot the mean line
    sns.boxplot(showmeans=True,
            meanline=True,
            meanprops={'color': 'gray', 'ls': '-', 'lw': 1},
            medianprops={'visible': False},
            whiskerprops={'visible': False},
            zorder=10,
            x="hist_dist",
            y="inst",
            data=probe_data,
            showfliers=False,
            showbox=False,
            showcaps=False,
            ax=ax2)
    
    #ax1.set_xlabel('Institution', fontsize=7)
    ax2.set_ylabel(None)
    ax2.set_xlim(0, 1500)
    ax2.set_xlabel('Histology distance (µm)')
    #ax2.get_legend().remove()
    #l = ax2.legend(fontsize=4, title='Advanced \n query', 
    #           title_fontsize=6, 
    #           loc='upper right',
    #           markerscale = 0.2 )
    #plt.setp(l.get_title(), multialignment='center')
    #leg.set_title('Passed adv. query',prop={'size':7})
    
    #ax1.tick_params(axis='x', labelrotation = 90)
    
    ax2.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax2.tick_params(axis='x', labelrotation=90)
    ax2.get_legend().remove()
    
    p_hist1 = permutation_test_histology(probe_data)
    p_hist2 = permutation_test_histology(probe_data)
    p_hist3 = permutation_test_histology(probe_data)
    p_hist = stat.mean([p_hist1, p_hist2, p_hist3])
    print("Histology PERMUTATION TEST ALL : ", p_hist)
    
    # permutation testing - PASS DATA ONLY
    probe_data_passed = probe_data[probe_data.eid.isin(passed_eids)]
    pp_h1 = permutation_test_histology(probe_data_passed)
    pp_h2 = permutation_test_histology(probe_data_passed)
    pp_h3 = permutation_test_histology(probe_data_passed)
    pp_h = stat.mean([pp_h1, pp_h2, pp_h3])
    print("Histology PERMUTATION TEST PASS : ", pp_h)
    
    ax1.set_title('Permutation Test p-value: \n    ALL : ' 
                    + str( round( p_hist, 4) )
                    + '    PASS : ' + str( round( pp_h, 4) ) )
    
    plt.tight_layout() # tighten layout around xlabel & ylabel
    
    fig.set_size_inches(2.15, 2.8)
    fig.savefig( str(Path(output, 'D_probe_dist_hist_all_lab.svg')), bbox_inches="tight" )
    #fig1.savefig( str(Path('figure_histology', 'probe-plots','micro_surf_err_plot.svg')), bbox_inches="tight" ) # tight ensures figure is in bounds of svg canvas!



def plot_probe_distance_histology_all(probe_data, output='figure_histology'):
    '''Plot the DISTANCES from planned to histology displacement, histology 
    track boxplott of ALL angles - to see its distribution shape.
    '''
    from pathlib import Path
    import os
    #from probe_geometry_analysis import probe_geometry_data as probe_geom_data
    import matplotlib.pyplot as plt
    import seaborn as sns
    #import numpy as np
    import reproducible_ephys_functions as ref
    import figure_hist_data as fhd
    import math
    
    ref.figure_style()
    
    # get probe data if necessary
    if 'probe_data' not in vars() or 'probe_data' not in globals():
        # get probe_data histology query
        probe_data = fhd.get_probe_data()
    
    # output DIR - generate output DIR for storing plots:
    OUTPUT = Path(output)
    if os.path.exists(OUTPUT) is False:
        os.mkdir(OUTPUT)
    
    # get institution map and colours
    lab_number_map, institution_map, institution_colors = ref.labs()
    
    # add institution col
    probe_data['inst'] = probe_data['lab'].map(institution_map)
    
    # create boxplot with institution colours
    #sns.boxplot(x='hist_error_surf', data=probe_data, 
    #            color = 'white',  orient="h",
    #            ax=ax1)
    #sns.stripplot(x='hist_error_surf', data=probe_data, 
    #              color = 'black', alpha = 0.8, size = 3, 
    #               orient="h",
    #              ax=ax1)
    # density plot
    x_dist = abs(abs(probe_data['hist_x']) - abs(probe_data['planned_x']))**2
    y_dist = abs(abs(probe_data['hist_y']) - abs(probe_data['planned_y']))**2
    sum_dist = x_dist + y_dist
    dist_list = []
    for s in sum_dist:
        dist_list.append(math.sqrt( s ) )
    
    probe_data['hist_dist'] = dist_list
    
    fig = sns.displot(probe_data['hist_dist'], kde=True)
    fig1 = fig.fig # fig is a FacetGrid object!
    ax1 = fig.ax # get axes
    ax1.set_xlim(0, 1500)
    #ax1.set_xlabel('Institution', fontsize=7)
    ax1.set_ylabel(None)
    #ax1.set_xlabel('Histology distance (µm)')
    ax1.set_xlabel(None)
    
    #ax1.tick_params(axis='x', labelrotation = 90)
    
    ax1.xaxis.set_major_locator(plt.MaxNLocator(5))
    #ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90, ha='right')
    ax1.tick_params(axis='x', labelrotation=90)
    #ax1.set(yticklabels=[])
    #ax1.tick_params(left=False)
    #ax1.set_axis_off()
    
    plt.tight_layout() # tighten layout around xlabel & ylabel
    
    fig1.set_size_inches(2.15, 1)
    fig1.savefig( str(Path(output, 'D_probe_dist_hist_all.svg')), bbox_inches="tight" )
    #fig1.savefig( str(Path('figure_histology', 'probe-plots','micro_surf_err_plot.svg')), bbox_inches="tight" ) # tight ensures figure is in bounds of svg canvas!




def plot_probe_distance_histology_lab(probe_data, output='figure_histology'):
    '''Plot the DISTANCES from planned to histology, histology track boxplots coloured
    by lab affiliation.  Add inter-lab means group to plot too?
    '''
    from pathlib import Path
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns
    import reproducible_ephys_functions as ref
    import figure_hist_data as fhd
    import math
    import statistics as stat
    
    # output DIR - generate output DIR for storing plots:
    OUTPUT = Path(output)
    if os.path.exists(OUTPUT) is False:
        os.mkdir(OUTPUT)
    
    # set figure style for consistency
    ref.figure_style()
    
    # get probe data if necessary
    if 'probe_data' not in vars() or 'probe_data' not in globals():
        # get probe_data histology query
        probe_data = fhd.get_probe_data()
    
    # get fig and axes
    fig1, ax1 = plt.subplots()
    
    # get institution map and colours
    lab_number_map, institution_map, institution_colors = ref.labs()
    
    # add institution col
    probe_data['inst'] = probe_data['lab'].map(institution_map)
    
    x_dist = abs(abs(probe_data['hist_x']) - abs(probe_data['planned_x']))**2
    y_dist = abs(abs(probe_data['hist_y']) - abs(probe_data['planned_y']))**2
    sum_dist = x_dist + y_dist
    dist_list = []
    for s in sum_dist:
        dist_list.append(math.sqrt( s ) )
    
    probe_data['hist_dist'] = dist_list
    
    # create boxplot with institution colours
    #sns.boxplot(y='inst', x='hist_dist', data=probe_data, 
    #            palette = institution_colors,  orient="h",
    #            ax=ax1)
    
    sns.stripplot(y='inst', x='hist_dist', data=probe_data, 
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
            x="hist_dist",
            y="inst",
            data=probe_data,
            showfliers=False,
            showbox=False,
            showcaps=False,
            ax=ax1)
    
    #ax1.set_xlabel('Institution', fontsize=7)
    ax1.set_ylabel(None)
    ax1.set_xlim(0, 1500)
    ax1.set_xlabel('Histology distance (µm)')
    
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
    fig1.savefig( str(Path(OUTPUT, 'D_probe_dist_hist_lab.svg')), bbox_inches="tight" )
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
    



def permutation_test_histology(probe_data, exclude_val = 4):
    '''
    Perform permutation test, excluding any groups with n below exclude_val.

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
    
    # get probe data if necessary
    if 'probe_data' not in vars() or 'probe_data' not in globals():
        # get probe_data histology query
        probe_data = fhd.get_probe_data()
    
    # get institution map and colours
    lab_number_map, institution_map, institution_colors = ref.labs()
    # add institution col
    probe_data['inst'] = probe_data['lab'].map(institution_map)
    
    # get histology distance
    x_dist = abs(abs(probe_data['hist_x']) - abs(probe_data['planned_x']))**2
    y_dist = abs(abs(probe_data['hist_y']) - abs(probe_data['planned_y']))**2
    sum_dist = x_dist + y_dist
    dist_list = []
    for s in sum_dist:
        dist_list.append(math.sqrt( s ) )
    
    probe_data['hist_dist'] = dist_list
    
    # exclude any rows that have less than 4 members in a group
    inst_counts = probe_data['inst'].value_counts()
    inst_exclude = []
    for i, name in enumerate(inst_counts.index):
        if inst_counts[i] < exclude_val:
            inst_exclude.append(name)
    for e in inst_exclude:
        index = probe_data[ probe_data['inst'] == e ].index
        probe_data.drop(index , inplace=True)
    
    # perform permutation test
    p_hist = permut_test( np.array( list(probe_data['hist_dist']) ),
                 metric=permut_dist,
                 labels1 = np.array( list(probe_data['lab']) ),
                 labels2 = np.array( list(probe_data['subject']) ) )
    
    return p_hist



def permutation_test_micro(probe_data, exclude_val = 4):
    '''
    Perform permutation test, excluding any groups with n below exclude_val.

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
    
    # get probe data if necessary
    if 'probe_data' not in vars() or 'probe_data' not in globals():
        # get probe_data histology query
        probe_data = fhd.get_probe_data()
    
    # get institution map and colours
    lab_number_map, institution_map, institution_colors = ref.labs()
    # add institution col
    probe_data['inst'] = probe_data['lab'].map(institution_map)
    
    # get histology distance
    x_dist = abs(abs(probe_data['micro_x']) - abs(probe_data['planned_x']))**2
    y_dist = abs(abs(probe_data['micro_y']) - abs(probe_data['planned_y']))**2
    sum_dist = x_dist + y_dist
    dist_list = []
    for s in sum_dist:
        dist_list.append(math.sqrt( s ) )
    
    probe_data['micro_dist'] = dist_list
    
    # exclude any rows that have less than 4 members in a group
    inst_counts = probe_data['inst'].value_counts()
    inst_exclude = []
    for i, name in enumerate(inst_counts.index):
        if inst_counts[i] < exclude_val:
            inst_exclude.append(name)
    for e in inst_exclude:
        index = probe_data[ probe_data['inst'] == e ].index
        probe_data.drop(index , inplace=True)
    
    # perform permutation test
    p_hist = permut_test( np.array( list(probe_data['micro_dist']) ),
                 metric=permut_dist,
                 labels1 = np.array( list(probe_data['lab']) ),
                 labels2 = np.array( list(probe_data['subject']) ) )
    
    return p_hist


def permutation_test_micro_old():
    '''
    

    Returns
    -------
    None.

    '''
    from pathlib import Path
    import matplotlib.pyplot as plt
    import numpy as np
    import figure_hist_data as fhd
    from permutation_test import permut_test
    import math
    
    # get probe data if necessary
    if 'probe_data' not in vars() or 'probe_data' not in globals():
        # get probe_data histology query
        probe_data = fhd.get_probe_data()
    
    # get micro distance
    x_dist = abs(abs(probe_data['micro_x']) - abs(probe_data['planned_x']))**2
    y_dist = abs(abs(probe_data['micro_y']) - abs(probe_data['planned_y']))**2
    sum_dist = x_dist + y_dist
    dist_list = []
    for s in sum_dist:
        dist_list.append(math.sqrt( s ) )
    
    probe_data['micro_dist'] = dist_list
    
    p_micro = permut_test( np.array( list(probe_data['micro_dist']) ),
                 metric=permut_dist,
                 labels1 = np.array( list(probe_data['lab']) ),
                 labels2 = np.array( list(probe_data['subject']) ) )
    
    return p_micro








def plot_probe_distance_histology_variance_labs(output='figure_histology'):
    '''Plot the MEAN DISTANCES from planned to histology by lab, histology 
    track boxplots coloured by lab affiliation.
    '''
    from pathlib import Path
    import os
    from probe_geometry_analysis import probe_geometry_data as probe_geom_data
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import reproducible_ephys_functions as ref
    
    # output DIR
    OUTPUT = Path(output)
    
    # generate output DIR for storing plots:
    if os.path.exists(OUTPUT) is False:
        os.mkdir(OUTPUT)
    
    # set figure style for consistency
    ref.figure_style()
    
    # load trajectory data for repeated site from local cache
    probe_data = probe_geom_data.load_trajectory_data('-2243_-2000')
    
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
    index = probe_data[ probe_data['subject'] == 'KS055' ].index
    probe_data.drop(index , inplace=True)
    index = probe_data[ probe_data['subject'] == 'NYU-27' ].index
    probe_data.drop(index , inplace=True)
    #ids.remove('CSH_ZAD_001') # this has very poor anatomy - dissection damage
    #ids.remove('CSHL051') # this has very poor anatomy - dissection damage
    #ids.remove('NYU-47') # it has no histology ???
    #ids.remove('UCLA011') # currently repeated site WAYYY OUT!
    #ids.remove('KS051') # this sample has not been imaged..
    # KS055 - error on surface is WAYYY OUT!
    # NYU-27 - error on surface is WAYYY OUT
    
    fig1, ax1 = plt.subplots()
    
    # get institution map and colours
    lab_number_map, institution_map, institution_colors = ref.labs()
    
    # add institution col
    probe_data['inst'] = probe_data['lab'].map(institution_map)
    
    # get standard deviation for each institution
    inst_sd = probe_data.groupby('inst')['hist_error_surf'].std()
    
    # and compute the inter-institution standard deviation
    mean_inst_sd = probe_data.groupby('inst')['hist_error_surf'].mean()
    inter_inst_sd = mean_inst_sd.std()
    
    sns.boxplot(y = inst_sd, color = 'white', ax=ax1)
    
    ax1.axhline(y = inter_inst_sd, color = 'red')
    
    sns.stripplot(y=inst_sd, alpha = 0.8, size = 3, color = 'black',
                  ax=ax1)
    
    #ax1.set_xlabel('Institution', fontsize=7)
    ax1.set_ylabel('Histology distance standard deviation', fontsize=7)
    
    ax1.yaxis.set_major_locator(plt.MaxNLocator(5))
    
    plt.tight_layout() # tighten layout around xlabel & ylabel
    
    fig1.set_size_inches(2, 2)
    fig1.savefig( str(Path(OUTPUT, 'D_probe_dist_hist_std.svg')), bbox_inches="tight" )
    #fig1.savefig( str(Path('figure_histology', 'probe-plots','micro_surf_err_plot.svg')), bbox_inches="tight" ) # tight ensures figure is in bounds of svg canvas!



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



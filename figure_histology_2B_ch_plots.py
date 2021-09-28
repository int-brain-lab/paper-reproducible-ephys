#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 17:08:49 2021

@author: sjwest
"""

from figure_histology import ch_disp_data as ch_data
from figure_histology import ch_disp_plots as ch_plots

import os
from pathlib import Path
import matplotlib.pyplot as plt

from one.api import ONE

from reproducible_ephys_functions import labs as labs_fun

one = ONE()

# download repeated site data
ch_data.download_ch_disp_data_rep_site()
 # and load it!
#channels_data = ch_data.load_ch_disp_data_rep_site()

# get all subject IDs:
ids = ch_plots.get_subj_IDs_rep_site()

# generate output DIR for storing plots:
if os.path.exists( Path('figure_histology', 'plots') ) is False:
    os.mkdir( Path('figure_histology', 'plots') )


 # generate and save coronal & sagittal figures for all ids if not generated
for i in range(0, len(ids)):
    
    
    if i<10:
        title = str('00'+str(i)+'_') # title precede with TWO 0s
    else:
        title = str('0'+str(i)+'_') # i is 10+, only ONE 0!
        
    # SKIP if coronal.svg EXISTS:
    if Path('figure_histology', 'plots', title+ids[i]+'_coronal.svg').exists():
        #print( i, ' ', ids[i] )
        print('')
    else: # process the subject:
        
        print( i, ' ', ids[i] )
        
        # plots of histology data along repeated site histology trajectory 
         # in coronal and sagittal planes
         
        try:
            # remove primary axis unless this is modulus 8 of index
            if i%8 == 0:
                plot = ch_plots.plot_atlas_traj(-2243, -2000, atlas_ID = ids[i],
                                provenance='Histology track')
            else:
                plot = ch_plots.plot_atlas_traj(-2243, -2000, atlas_ID = ids[i],
                                provenance='Histology track', remove_primary_axis = True)
            
            
            
            # plot the channels on the axis
             # colours : deepskyblue b g r w k y m c deepskyblue
            plot = ch_plots.plot_subj_channels(plot, colour = 'w')
            
            if i<10:
                title = str('00'+str(i)+'_') # title precede with TWO 0s
            else:
                title = str('0'+str(i)+'_') # i is 10+, only ONE 0!
            
            plot['cax'].savefig( str(Path('figure_histology', 'plots', title+ids[i]+'_coronal.svg')), bbox_inches="tight" ) 
              # tight ensures figure is in bounds of svg canvas!
            
            plot['sax'].savefig( str(Path('figure_histology', 'plots', title+ids[i]+'_sagittal.svg')), bbox_inches="tight" ) 
              # tight ensures figure is in bounds of svg canvas!
            
            plt.close(plot['cax'])
            plt.close(plot['sax']) # close all plt windows at end of each for loop to prevent memory leak via pyplot state machine
              # pyplot state machine holds onto references to figures and axes even if user reference is closed!
        except Exception:
            print("ERROR - no track?")
        
        


# NEXT - generate a large matplotlib canvas for all axes


fig, axs = plt.subplots(2, 2)

# get all lab ids for subject ids:
labs = []
for s in range(0, len(ids) ):
    l = one.alyx.rest('insertions', 'list', subject = ids[s])[0]['session_info']['lab']
    labs.append(l)

 # get number and colour maps for labs
lab_number_map, institution_map, institution_colors = labs_fun()




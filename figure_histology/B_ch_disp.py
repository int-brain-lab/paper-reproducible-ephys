#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 17:08:49 2021

@author: sjwest
"""

import ch_disp_data as ch_data
import ch_disp_plots as ch_plots

import os
from pathlib import Path

import matplotlib.pyplot as plt

# download repeated site data
ch_data.download_ch_disp_data_rep_site()
 # and load it!
#channels_data = ch_data.load_ch_disp_data_rep_site()

# get all subject IDs:
ids = ch_plots.get_subj_IDs_rep_site()

# generate output DIR for storing plots:
if os.path.exists('plots') is False:
    os.mkdir('plots')


 # generate and save coronal & sagittal figures for all ids 
for i in range(0, len(ids)):
    
    print( i, ' ', ids[i] )
    
    # plots of histology data along repeated site histology trajectory 
     # in coronal and sagittal planes
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
    
    plot['cax'].savefig( str(Path('plots', title+ids[i]+'_coronal.svg')), bbox_inches="tight" ) 
      # tight ensures figure is in bounds of svg canvas!
    
    plot['sax'].savefig( str(Path('plots', title+ids[i]+'_sagittal.svg')), bbox_inches="tight" ) 
      # tight ensures figure is in bounds of svg canvas!
    
    plt.close(plot['cax'])
    plt.close(plot['sax']) # close all plt windows at end of each for loop to prevent memory leak via pyplot state machine
      # pyplot state machine holds onto references to figures and axes even if user reference is closed!
    







#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 17:08:49 2021

Generates a matrix of plots of all repeated site insertions

* 

@author: sjwest
"""

from figure_histology import ch_disp_data as ch_data
from figure_histology import ch_disp_plots as ch_plots

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


# save data relating subject/lab to institute and colour
dat = pd.DataFrame()
dat['subject'] = ids

# get all lab ids for subject ids:
labs = []
for s in range(0, len(ids) ):
    l = one.alyx.rest('insertions', 'list', subject = ids[s])[0]['session_info']['lab']
    labs.append(l)

dat['lab'] = labs

 # get number and colour maps for labs in list
lab_number_map, institution_map, institution_colors = labs_fun()

dat['lab_number']  = [lab_number_map[k] for k in labs]
dat['institute']  = [institution_map[k] for k in labs]
dat['institute_colour'] = [institution_colors[k] for k in [institution_map[k] for k in labs]]
dat = dat.sort_values(by=['institute', 'subject']).reset_index(drop=True)
rec_per_lab = dat.groupby('institute').size()
dat['recording'] = np.concatenate([np.arange(i) for i in rec_per_lab.values])


 # generate a figure with axes - 1 row per institute, 1 col per subject
  # coronal and sagittal
figc, axsc = plt.subplots( len(rec_per_lab), max(rec_per_lab))
figs, axss = plt.subplots( len(rec_per_lab), max(rec_per_lab))


figc = plt.figure()
figs = plt.figure()

for i in range(0, len(rec_per_lab)):
    for j in range((rec_per_lab[i]), max(rec_per_lab)):
        print('i j : ', i, j)
        figc.delaxes(axsc[i][j])
        figs.delaxes(axss[i][j])


 # load each image into the corresponding axis in fig

inst_index = -1

#for i in range(0, 2:
for i in range(0, len(dat)):
    
    print(i)
    
    try:
        if dat['recording'][i] == 0:
            inst_index = inst_index+1
            
            print( 'inst ', str(inst_index) )
            plot = ch_plots.plot_atlas_traj(-2243, -2000, atlas_ID = ids[i],
                                    provenance='Histology track',
                                    axc = axsc[inst_index, dat['recording'][i]],
                                    axs = axss[inst_index, dat['recording'][i]] )
        else:
            plot = ch_plots.plot_atlas_traj(-2243, -2000, atlas_ID = ids[i],
                                    provenance='Histology track', 
                                    remove_primary_axis = True,
                                    axc = axsc[inst_index, dat['recording'][i]],
                                    axs = axss[inst_index, dat['recording'][i]])
        
        # plot the channels on the axis - white
        plot = ch_plots.plot_subj_channels(plot, colour = 'w')
        
        #print( 'subj ', str(dat['recording'][i]) )
        #axsc[inst_index, dat['recording'][i]] = plot['cax'].axes
        #axss[inst_index, dat['recording'][i]] = plot['sax'].axes
    except Exception:
            print("ERROR - no track..")
    


figc.delaxes(axsc[6][10])
figs.delaxes(axss[6][10])

wspace = 0.5   # the amount of width reserved for blank space between subplots
  # 0.5 gives the tightest layout without overlap between subplots
hspace = 0.1   # the amount of height reserved for white space between subplots

figc.subplots_adjust(wspace, hspace)
figs.subplots_adjust(wspace, hspace)


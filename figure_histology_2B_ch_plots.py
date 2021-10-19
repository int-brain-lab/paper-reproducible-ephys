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
import numpy as np
import pandas as pd

from one.api import ONE

from reproducible_ephys_functions import labs as labs_fun

one = ONE()

### OUTPUT DIR ###
OUTPUT = Path('figure_histology', 'plots')

# generate output DIR for storing plots - relative to script location
if os.path.exists( OUTPUT ) is False:
    os.mkdir( OUTPUT )

# download repeated site data
ch_data.download_ch_disp_data_rep_site()
 # and load it!
#channels_data = ch_data.load_ch_disp_data_rep_site()

# get all subject IDs:
ids = ch_plots.get_subj_IDs_rep_site()

ids.remove('NYU-47') # it has no histology ???

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


# len(dat) - is currently 66

 # generate a figure with axes - 4 rows, 17 cols (max 68)
  # coronal and sagittal
nRows = 4
nCols = 17
figc, axsc = plt.subplots( nRows, nCols)
figs, axss = plt.subplots( nRows, nCols)

 # remove excess axes 
last_count = (nRows*nCols) - len(dat) - 1
#last_count = len(dat) -  (nRows*nCols) 
for i in range( nCols-last_count-1, nCols):
    print('col : ', i)
    figc.delaxes( axsc[(nRows-1)][i] )
    figs.delaxes( axss[(nRows-1)][i] )

 # load each image into the corresponding axis in fig

row_index = -1
col_index = 0
#for i in range(0, 2:
for i in range(0, len(dat)):
    
    print(i)
    if(i%nCols == 0):
        row_index=row_index+1
    col_index = i%nCols
    
    print('row ', row_index)
    print('col', col_index)
    
    # generate plots of slice along the repeated site histology trajectory
     # this will download the histology images if necessary from flatiron
    try:
        if col_index == 0:
            
            plot = ch_plots.plot_atlas_traj(-2243, -2000, atlas_ID = ids[i],
                                    provenance='Histology track',
                                    axc = axsc[row_index, col_index],
                                    axs = axss[row_index, col_index] )
        else:
            plot = ch_plots.plot_atlas_traj(-2243, -2000, atlas_ID = ids[i],
                                    provenance='Histology track', 
                                    remove_primary_axis = True,
                                    axc = axsc[row_index, col_index],
                                    axs = axss[row_index, col_index])
        
        # plot the channels on the axis - white
        plot = ch_plots.plot_subj_channels(plot, colour = 'w')
        
        #print( 'subj ', str(col_index) )
        #axsc[row_index, col_index] = plot['cax'].axes
        #axss[row_index, col_index] = plot['sax'].axes
    except Exception:
            print("ERROR - no track. Index ", i)
    


# make plots BIG
figc.set_size_inches(60, 20)
figs.set_size_inches(60, 20)

# adjust spacing
wspace = 0.3   # the amount of width reserved for blank space between subplots
  # gives the tightest layout without overlap between subplots
hspace = 0.1   # the amount of height reserved for white space between subplots

figc.subplots_adjust(wspace, hspace)
figs.subplots_adjust(wspace, hspace)

# save to output
figc.savefig( str(Path(OUTPUT, 'ALL_coronal.svg')), bbox_inches="tight" )
figs.savefig( str(Path(OUTPUT, 'ALL_sagittal.svg')), bbox_inches="tight" )

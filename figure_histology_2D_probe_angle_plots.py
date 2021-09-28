#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 17:08:49 2021

@author: sjwest
"""


from one.api import ONE
import ibllib.atlas as atlas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date

import os
from pathlib import Path

import probe_disp_data as pr_data
 #import ch_disp_from_planned_plots as ch_plots

# download repeated site data
probe_data = pr_data.download_probe_disp_data_rep_site()

probe_data = pr_data.load_probe_disp_data_rep_site()


# generate output DIR for storing plots:
if os.path.exists( Path('figure_histology', 'probe-plots') ) is False:
    os.mkdir( Path('figure_histology', 'probe-plots') )



################################################################################

### PLOT SURFACE ERRORS ###
 # 
 # Plot displacement between probe entry coordinates:
 # 
 #  Planned : to Micro-Manipulator, to Histology Track
 # 
 #    This will show the absolute error estimated by micro-manipulator, and recorded
 #     by the actual histology.
 # 
 #  Micro-Manipulator : to Histology Track
 # 
 #     This will show the error between the estimated placement from micro-manipulator
 #      and recorded by the actual histology.
 # 

################################################################################


# connect to ONE
one = ONE()

# get new atlas for plotting:
brain_atlas = atlas.AllenAtlas(res_um=25)


################################################################################

### MICRO-MANIPULATOR Surface Insertion Coord plot

# generate matplotlib pyplot figs/axes 
fig1, ax1 = plt.subplots()

#ax3.scatter( probe_data['hist_y'], probe_data['hist_x'], c='deepskyblue', s=1)

# empty numpy arrays for storing the entry point of probe into brain
 # and "exit point" i.e the probe tip!
all_ins_entry = np.empty((0, 3))

# plot line and points from planned insertion entry to actual histology entry
for idx in range(len(probe_data)):
    all_ins_entry = np.vstack([all_ins_entry, 
                               np.array( (probe_data['micro_x'][idx]/1e6, 
                                          probe_data['micro_y'][idx]/1e6, 
                                          probe_data['micro_z'][idx]/1e6) )  ])
    
    ax1.plot( [probe_data['micro_y'][idx], probe_data['planned_y'][0] ], 
          [probe_data['micro_x'][idx], probe_data['planned_x'][0] ], 
          color='green', marker="o", markersize=1, linewidth = 0.2 )


# plot the planned insertion entry as large blue dot
ax1.plot(probe_data['planned_y'][0], probe_data['planned_x'][0], 
         color='darkblue', marker="o", markersize=3)


# Compute targeting error at surface of brain
error_top = all_ins_entry - np.array( (probe_data['planned_x'][0]/1e6, 
                                          probe_data['planned_y'][0]/1e6, 
                                          probe_data['planned_z'][0]/1e6) )
distance_top = np.sqrt(np.sum(error_top ** 2, axis=1)) # distance between xyz coords
top_mean = np.mean(distance_top)*1e6
top_std = np.std(distance_top)*1e6

rms_top = np.sqrt(np.mean(distance_top ** 2))*1e6

# set x/y axis labels
ax1.set_xlabel('ap (um)', fontsize=7)
ax1.tick_params(axis='x', labelsize=7)
ax1.set_ylabel('ml (um)', fontsize=7)
ax1.tick_params(axis='y', labelsize=7)
ax1.yaxis.set_major_locator(plt.MaxNLocator(4))

#ax1.set_ylim((-3000,-1250))
#ax1.set_xlim((-1000,-3500))
ax1.set_ylim((-3000,-500))
ax1.set_xlim((-500,-3500))

plt.tight_layout() # tighten layout around xlabel & ylabel

fig1.set_size_inches(3, 3)
fig1.savefig( str(Path('figure_histology', 'probe-plots','micro_surf_err_plot.svg')), bbox_inches="tight" ) # tight ensures figure is in bounds of svg canvas!

# add mean trageting error distance to title
ax1.set_title('EXPERIMENTER: Mean distance ' +
              str(np.around(top_mean, 1)) + ' µm', fontsize=10)

fig1.savefig( str(Path('figure_histology', 'probe-plots','micromanipulator_surface_error_plot.svg')), bbox_inches="tight" ) # tight ensures figure is in bounds of svg canvas!

plt.show()



# boxplot
bfig3, bax3 = plt.subplots()

distance_m_top_um = distance_top * 1e6

bax3.boxplot(distance_m_top_um, patch_artist=True,
             boxprops=dict(facecolor='white'),
            medianprops=dict(color='green') )

bax3.set_xlabel('', fontsize=7)
bax3.tick_params(axis='x', labelsize=7)
bax3.set_ylabel('Experimenter distance (µm)', fontsize=7)
bax3.tick_params(axis='y', labelsize=7)
bax3.set_ylim((0,1400))
#bax3.set_xlim((-1000,-3500))

plt.tight_layout() # tighten layout around xlabel & ylabel

bfig3.set_size_inches(3, 3)
bfig3.savefig( str(Path('figure_histology', 'probe-plots','micro_dist_box.svg')), bbox_inches="tight" ) # tight ensures figure is in bounds of svg canvas!

bax3.set_title('EXPERIMENTER: Mean Distance ' +
              str(np.around(top_mean, 1)) + ' µm', fontsize=10)
#bax3.yaxis.set_major_locator(plt.MaxNLocator(4))

bfig3.set_size_inches(3, 3)
bfig3.savefig( str(Path('figure_histology', 'probe-plots','micromanipulator_distance_boxplot.svg')), bbox_inches="tight" ) # tight ensures figure is in bounds of svg canvas!

plt.show()


################################################################################

### HISTOLOGY Surface Insertion Coord plot

# generate matplotlib pyplot figs/axes 
fig2, ax2 = plt.subplots()


#ax3.scatter( probe_data['hist_y'], probe_data['hist_x'], c='deepskyblue', s=1)

# empty numpy arrays for storing the entry point of probe into brain
# and "exit point" i.e the probe tip!
all_ins_entry = np.empty((0, 3))

# plot line and points from planned insertion entry to actual histology entry
for idx in range(len(probe_data)):
    all_ins_entry = np.vstack([all_ins_entry, 
                               np.array( (probe_data['hist_x'][idx]/1e6, 
                                          probe_data['hist_y'][idx]/1e6, 
                                          probe_data['hist_z'][idx]/1e6) )  ])
    
    ax2.plot( [probe_data['hist_y'][idx], probe_data['planned_y'][0] ], 
          [probe_data['hist_x'][idx], probe_data['planned_x'][0] ], 
          color='orangered', marker="o", markersize=1, linewidth = 0.2 )


# plot the planned insertion entry as large blue dot
ax2.plot(probe_data['planned_y'][0], probe_data['planned_x'][0], 
         color='darkblue', marker="o", markersize=3)


# Compute targeting error at surface of brain
error_top = all_ins_entry - np.array( (probe_data['planned_x'][idx]/1e6, 
                                          probe_data['planned_y'][idx]/1e6, 
                                          probe_data['planned_z'][idx]/1e6) )
distance_top = np.sqrt(np.sum(error_top ** 2, axis=1))
top_mean = np.mean(distance_top)*1e6
top_std = np.std(distance_top)*1e6

rms_top = np.sqrt(np.mean(distance_top ** 2))*1e6

ax2.set_xlabel('ap (um)', fontsize=7)
ax2.tick_params(axis='x', labelsize=7)
ax2.set_ylabel('ml (um)', fontsize=7)
ax2.tick_params(axis='y', labelsize=7)
ax2.yaxis.set_major_locator(plt.MaxNLocator(4))

ax2.set_ylim((-3000,-500))
ax2.set_xlim((-500,-3500))

plt.tight_layout() # tighten layout around xlabel & ylabel

fig2.set_size_inches(3, 3)
fig2.savefig( str(Path('figure_histology', 'probe-plots','hist_surf_err_plot.svg')), bbox_inches="tight" ) # tight ensures figure is in bounds of svg canvas!

ax2.set_title('HISTOLOGY: Mean distance ' +
              str(np.around(top_mean, 1)) + ' µm', fontsize=10)

fig2.savefig( str(Path('figure_histology', 'probe-plots','histology_surface_error_plot.svg')), bbox_inches="tight" ) # tight ensures figure is in bounds of svg canvas!

plt.show()



# boxplot

bfig3, bax3 = plt.subplots()

distance_h_top_um = distance_top * 1e6

bax3.boxplot(distance_h_top_um, patch_artist=True,
             boxprops=dict(facecolor='white'),
            medianprops=dict(color='orangered') )

bax3.set_xlabel('', fontsize=7)
bax3.tick_params(axis='x', labelsize=7)
bax3.set_ylabel('Histology distance (µm)', fontsize=7)
bax3.tick_params(axis='y', labelsize=7)
bax3.set_ylim((0,1400))
#bax3.set_xlim((-1000,-3500))

plt.tight_layout() # tighten layout around xlabel & ylabel

bfig3.set_size_inches(3, 3)
bfig3.savefig( str(Path('figure_histology', 'probe-plots','hist_dist_box.svg')), bbox_inches="tight" ) # tight ensures figure is in bounds of svg canvas!

bax3.set_title('HISTOLOGY: Mean Distance ' +
              str(np.around(top_mean, 1)) + ' µm', fontsize=10)
#bax3.yaxis.set_major_locator(plt.MaxNLocator(4))

bfig3.savefig( str(Path('figure_histology', 'probe-plots','histology_distance_boxplot.svg')), bbox_inches="tight" ) # tight ensures figure is in bounds of svg canvas!

plt.show()


################################################################################

### MICROMANIPULATOR - HISTOLOGY Surface Insertion Coord plot

# generate matplotlib pyplot figs/axes 
fig3, ax3 = plt.subplots()


#ax3.scatter( probe_data['hist_y'], probe_data['hist_x'], c='deepskyblue', s=1)

error_top = np.empty((0, 3))

# plot line and points from planned insertion entry to actual histology entry
for idx in range(len(probe_data)):
    
    # compute error from histology xyz to micro xyz
    error_top = np.vstack([error_top, 
                           np.array( (probe_data['hist_x'][idx]/1e6, probe_data['hist_y'][idx]/1e6, 
         probe_data['hist_z'][idx]/1e6) ) - np.array( (probe_data['micro_x'][idx]/1e6, 
                                          probe_data['micro_y'][idx]/1e6, probe_data['micro_z'][idx]/1e6) ) ])
    
    ax3.plot( [probe_data['hist_y'][idx], probe_data['micro_y'][idx] ], 
          [probe_data['hist_x'][idx], probe_data['micro_x'][idx] ], 
          color='black', linewidth = 0.2 )
    ax3.scatter(probe_data['micro_y'][idx], probe_data['micro_x'][idx],
                c='green', s=2)
    ax3.scatter(probe_data['hist_y'][idx], probe_data['hist_x'][idx],
                c='orangered', s=2)


# plot the planned insertion entry as large blue dot
ax3.plot(probe_data['planned_y'][0], probe_data['planned_x'][0], 
         color='darkblue', marker="o", markersize=3)


# Compute targeting error at surface of brain
#error_top = all_ins_entry - ins_plan.xyz[0, :]
distance_top = np.sqrt(np.sum(error_top ** 2, axis=1)) # this computes the DISTANCE between xyz coords!
top_mean = np.mean(distance_top)*1e6
top_std = np.std(distance_top)*1e6

 # root mean squared of the distances from micro to hist
rms_top = np.sqrt(np.mean(distance_top ** 2))*1e6

ax3.set_xlabel('ap (um)', fontsize=7)
ax3.tick_params(axis='x', labelsize=7)
ax3.set_ylabel('ml (um)', fontsize=7)
ax3.tick_params(axis='y', labelsize=7)

ax3.yaxis.set_major_locator(plt.MaxNLocator(4))

#ax3.set_ylim((-3000,-1250))
#ax3.set_xlim((-1000,-3500))
ax3.set_ylim((-3000,-500))
ax3.set_xlim((-500,-3500))

plt.tight_layout() # tighten layout around xlabel & ylabel

fig3.set_size_inches(3, 3)
fig3.savefig( str(Path('figure_histology', 'probe-plots','micro-_to_hist_surf_err_plot.svg')), bbox_inches="tight" ) # tight ensures figure is in bounds of svg canvas!

ax3.set_title('EXPERIMENTER to HISTOLOGY: Mean distance ' +
              str(np.around(top_mean, 1)) + ' µm', fontsize=10)

fig3.savefig( str(Path('figure_histology', 'probe-plots','micro-_to_histology_surface_error_plot.svg')), bbox_inches="tight" ) # tight ensures figure is in bounds of svg canvas!

plt.show()


# boxplot

bfig3, bax3 = plt.subplots()

distance_top_um = distance_top * 1e6

bax3.boxplot(distance_top_um, patch_artist=True,
             boxprops=dict(facecolor='white'),
            medianprops=dict(color='black') )

bax3.set_xlabel('', fontsize=7)
bax3.tick_params(axis='x', labelsize=7)
bax3.set_ylabel('Experimenter - Histology distance (µm)', fontsize=7)
bax3.tick_params(axis='y', labelsize=7)


bax3.set_ylim((0,1400))
#bax3.set_xlim((-1000,-3500))

plt.tight_layout() # tighten layout around xlabel & ylabel

bfig3.set_size_inches(3, 3)
bfig3.savefig( str(Path('figure_histology', 'probe-plots','micro-_to_hist_dist_box.svg')), bbox_inches="tight" ) # tight ensures figure is in bounds of svg canvas!

bax3.set_title('EXPERIMENTER to HISTOLOGY: Mean Distance ' +
              str(np.around(top_mean, 1)) + ' µm', fontsize=10)
#bax3.yaxis.set_major_locator(plt.MaxNLocator(4))

bfig3.savefig( str(Path('figure_histology', 'probe-plots','micro-_to_histology_distance_boxplot.svg')), bbox_inches="tight" ) # tight ensures figure is in bounds of svg canvas!

plt.show()



################################################################################

### Surface Insertion Coord Scatterplot:
    # MICROMANIPULATOR to PLANNED Distance against HISTOLOGY to MICROMANIPULATOR Distance

# generate matplotlib pyplot figs/axes 
sfig, sax = plt.subplots()

# OMIT UCLA005 & UCLA011 as there is an error in micro coords for these
 # np.argwhere(distance_m_top_um > 3000) # returns indices 66,67,68,69
 # probe_data['subject'][66] # and 67,68,69 return UCLA005 UCLA011
  # OMIT the offending indices from both distance measures - i.e concat 1:65 AND 70:72
distance_m_p_top_um_2 = np.concatenate( [ distance_m_top_um[1:65], distance_m_top_um[70:72] ] )
distance_h_p_top_um_2 = np.concatenate( [ distance_h_top_um[1:65], distance_h_top_um[70:72] ] )

distance_h_m_top_um_2 = np.concatenate( [ distance_top_um[1:65], distance_top_um[70:72] ] )

#sax.scatter(distance_h_top_um, distance_m_top_um, c='k', marker ='.', s=1)
sax.scatter(distance_h_m_top_um_2, distance_m_p_top_um_2, c='k', marker ='.', s=1)
#sax.scatter(distance_h_p_top_um_2, distance_m_p_top_um_2, c='k', marker ='.', s=1)

sax.set_xlabel('Micro-Manipulator to Histology distance (µm)', fontsize=7)
sax.tick_params(axis='x', labelsize=7)
sax.set_ylabel('Planned to Micro-Manipulator distance (µm)', fontsize=7)
sax.tick_params(axis='y', labelsize=7)

plt.tight_layout() # tighten layout around xlabel & ylabel

sfig.set_size_inches(3, 3)
sfig.savefig( str(Path('figure_histology', 'probe-plots','micro-planned_to_hist-micro_dist_scatter.svg')), bbox_inches="tight" ) # tight ensures figure is in bounds of svg canvas!

plt.show()




################################################################################

### Surface Insertion Coord Scatterplot:
    # MICROMANIPULATOR to PLANNED Distance against HISTOLOGY to PLANNED Distance

# generate matplotlib pyplot figs/axes 
sfig, sax = plt.subplots()

# OMIT UCLA005 & UCLA011 as there is an error in micro coords for these
 # np.argwhere(distance_m_top_um > 3000) # returns indices 66,67,68,69
 # probe_data['subject'][66] # and 67,68,69 return UCLA005 UCLA011
  # OMIT the offending indices from both distance measures - i.e concat 1:65 AND 70:72
distance_m_p_top_um_2 = np.concatenate( [ distance_m_top_um[1:65], distance_m_top_um[70:72] ] )
distance_h_p_top_um_2 = np.concatenate( [ distance_h_top_um[1:65], distance_h_top_um[70:72] ] )

distance_h_m_top_um_2 = np.concatenate( [ distance_top_um[1:65], distance_top_um[70:72] ] )

#sax.scatter(distance_h_top_um, distance_m_top_um, c='k', marker ='.', s=1)
#sax.scatter(distance_h_m_top_um_2, distance_m_p_top_um_2, c='k', marker ='.', s=1)
sax.scatter(distance_h_p_top_um_2, distance_m_p_top_um_2, c='k', marker ='.', s=1)

sax.set_xlabel('Planned to Histology distance (µm)', fontsize=7)
sax.tick_params(axis='x', labelsize=7)
sax.set_ylabel('Planned to Micro-Manipulator distance (µm)', fontsize=7)
sax.tick_params(axis='y', labelsize=7)

plt.tight_layout() # tighten layout around xlabel & ylabel

sfig.set_size_inches(3, 3)
sfig.savefig( str(Path('figure_histology', 'probe-plots','micro-planned_to_hist-planned_dist_scatter.svg')), bbox_inches="tight" ) # tight ensures figure is in bounds of svg canvas!

plt.show()




"""
Get targeting accuracy and number of neurons per region for all repeated site recordings

Created on Wed Sep  9 09:15:34 2020
Guido Meijer
"""

from os.path import join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import alf
import brainbox.io.one as bbone
from reproducible_ephys_functions import query
from oneibl.one import ONE
one = ONE()

# List of regions in repeated site
target_regions = ['VISa1', 'VISa2/3', 'VISa4', 'VISa5', 'VISa6a', 'VISa6b', 'CA1', 'DG-mo', 'DG-sg',
                  'DG-po', 'CA3', 'LP', 'PO']

# Create empty dataframes
hit_regions = pd.DataFrame(columns=target_regions)
cell_counts = pd.DataFrame(columns=target_regions)
other_recs = pd.DataFrame(columns=target_regions)

# Query repeated site sessions
rep_site = query()

for i in range(len(rep_site)):
    print('Processing session %d of %d' % (i+1, len(rep_site)))
    
    # Load in data
    eid = rep_site[i]['session']['id']
    try:
        spikes, clusters, channels = bbone.load_spike_sorting_with_channel(eid, one=one)
    except:
        continue
       
    # Get coordinates of micro-manipulator and histology
    hist = one.alyx.rest('trajectories', 'list', provenance='Histology track',
                     probe_insertion=rep_site[i]['probe_insertion'])
    if len(hist) == 0:
        continue
    hit_regions.loc[eid, 'x_hist'] = hist[0]['x']
    hit_regions.loc[eid, 'y_hist'] = hist[0]['y']
    manipulator = one.alyx.rest('trajectories', 'list', provenance='Micro-manipulator',
                                probe_insertion=rep_site[i]['probe_insertion'])
    if len(manipulator) > 0:
        hit_regions.loc[eid, 'x_target'] = manipulator[0]['x']
        hit_regions.loc[eid, 'y_target'] = manipulator[0]['y']
    
    # Check if recorded regions are in target regions   
    rec_regions = channels[rep_site[i]['probe_name']]['acronym']
    this_hits = np.zeros(len(target_regions))
    this_hits[[j for j, k in enumerate(target_regions) if k in rec_regions]] = 1
    hit_regions.loc[eid, target_regions] = this_hits
    
    # Get number of neurons per region
    clus_regions = list(clusters[rep_site[i]['probe_name']]['acronym'])
    for j, region in enumerate(target_regions):
        cell_counts.loc[eid, region] = clus_regions.count(region)
   
# Calculate distance to target at surface
hit_regions['dist'] = np.sqrt((hit_regions['x_hist'] - -2243)**2
                              + (hit_regions['y_hist'] - -2000)**2)

# Query other sessions that hit the regions of the repeated site
for i, region in enumerate(target_regions):
    ses = one.alyx.rest('sessions', 'list', atlas_acronym=region, histology=True,
                        django='project__name__icontains,ibl_neuropixel_brainwide_01')
    other_recs.loc[0, region] = len(ses) - hit_regions[region].sum()
    
# %% Plot

dist_cutoff = 300
region_colors = [(0, 128/255, 129/255), (0, 128/255, 129/255), (0, 128/255, 129/255),
                 (0, 128/255, 129/255), (0, 128/255, 129/255), (0, 128/255, 129/255),
                 (76/255, 187/255, 23/255), (76/255, 187/255, 23/255), (76/255, 187/255, 23/255),
                 (76/255, 187/255, 23/255), (76/255, 187/255, 23/255), (254/255, 127/255, 156/255),
                 (254/255, 127/255, 156/255)] 

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 12))
sns.set(style="ticks", context="paper", font_scale=1.7)

ax1.scatter(hit_regions['x_hist'], hit_regions['y_hist'], s=50)
ax1.scatter(-2243, -2000, color='k', s=50)
ax1.set(ylim=[-3000, -1500], xlim=[-3000, -1500], yticks=[-3000, -2500, -2000, -1500],
        xlabel='ML coordinates (\u03BCm)', ylabel='AP coordinates (\u03BCm)')

ax2.barh(target_regions, hit_regions[target_regions].mean() * 100)
ax2.set(xlabel='Repeated site targeting accuracy (%)',
        title='n = %d recordings' % hit_regions.shape[0],
        xlim=[0, 100])
ax2.invert_yaxis()  

incl_rec = hit_regions['dist'] < dist_cutoff
ax3.add_artist(plt.Circle((-2243, -2000), dist_cutoff, color=[0.6, 0.6, 0.6], linestyle='--', lw=3,
                          fill=False))
ax3.scatter(hit_regions['x_hist'], hit_regions['y_hist'], color='r', s=50)
ax3.scatter(hit_regions.loc[incl_rec, 'x_hist'], hit_regions.loc[incl_rec, 'y_hist'],
            color='g', s=50)
ax3.scatter(-2243, -2000, color='k', s=50)
ax3.set(ylim=[-3000, -1500], xlim=[-3000, -1500], yticks=[-3000, -2500, -2000, -1500],
        xlabel='ML coordinates (\u03BCm)', ylabel='AP coordinates (\u03BCm)')

ax4.barh(target_regions, hit_regions.loc[incl_rec, target_regions].mean() * 100)
ax4.set(xlabel='Repeated site targeting accuracy (%)',
        title='n = %d recordings' % np.sum(incl_rec),
        xlim=[0, 100])
ax4.invert_yaxis()  

plt.tight_layout(pad=2.5)
sns.despine(trim=False)

f, ax1 = plt.subplots(1, 1, figsize=(12, 6))
sns.set(style="ticks", context="paper", font_scale=1.7)
sns.swarmplot(data=cell_counts, palette=region_colors)
ax1.set(ylabel='Number of cells per recording')

plt.tight_layout(pad=2.5)
sns.despine(trim=True)

f, ax1 = plt.subplots(1, 1, figsize=(12, 6))
sns.set(style="ticks", context="paper", font_scale=1.7)
sns.swarmplot(data=cell_counts, palette=region_colors)
ax1.set(ylabel='Number of cells per recording')

plt.tight_layout(pad=2.5)
sns.despine(trim=True)


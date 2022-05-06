"""
Investigate the variability between different labs

Created on Fri Sep 25 11:26:39 2020
Guido Meijer
"""

import numpy as np
import pandas as pd
import seaborn as sns
from os.path import join
import matplotlib.pyplot as plt
import brainbox.io.one as bbone
from reproducible_ephys_functions import query, labs
from reproducible_ephys_paths import FIG_PATH
from oneibl.one import ONE
one = ONE()

# Query repeated site trajectories
traj = query()

# Initialize dataframe
rep_site = pd.DataFrame()

# %% Loop through repeated site recordings and extract the data
for i in range(len(traj)):
    print('Processing repeated site recording %d of %d' % (i+1, len(traj)))
    
    # Load in data
    eid = traj[i]['session']['id']
    try:
        spikes, clusters, channels = bbone.load_spike_sorting_with_channel(eid, one=one)
    except:
        continue
       
    # Get coordinates of micro-manipulator and histology
    hist = one.alyx.rest('trajectories', 'list', provenance='Histology track',
                     probe_insertion=traj[i]['probe_insertion'])
    if len(hist) == 0:
        continue
    rep_site.loc[i, 'x_hist'] = hist[0]['x']
    rep_site.loc[i, 'y_hist'] = hist[0]['y']
    manipulator = one.alyx.rest('trajectories', 'list', provenance='Micro-manipulator',
                                probe_insertion=traj[i]['probe_insertion'])
    if len(manipulator) > 0:
        rep_site.loc[i, 'x_target'] = manipulator[0]['x']
        rep_site.loc[i, 'y_target'] = manipulator[0]['y']
        
    # Get number of units
    probe = traj[i]['probe_name']
    rep_site.loc[i, 'n_good'] = np.sum(clusters[probe]['metrics']['ks2_label'] == 'good')
    rep_site.loc[i, 'n_mua'] = np.sum(clusters[probe]['metrics']['ks2_label'] == 'mua')
    
    # Get firing rates
    fr = []
    good_clusters = clusters[probe]['metrics']['cluster_id'][
                                        clusters[probe]['metrics']['ks2_label'] == 'good']
    for j, cluster in enumerate(good_clusters):
        fr.append(spikes[probe]['times'][spikes[probe]['clusters'] == cluster].shape[0]
                  / spikes[probe]['times'][-1])
    rep_site.loc[i, 'median_fr'] = np.median(fr)
    
    # Get session data
    rep_site.loc[i, 'eid'] = eid
    rep_site.loc[i, 'probe'] = probe
    rep_site.loc[i, 'lab'] = traj[i]['session']['lab']
    rep_site.loc[i, 'subject'] = traj[i]['session']['subject']
    
# Calculate distance from target
rep_site['targeting_error'] = np.sqrt(((rep_site['x_hist'] - -2243) ** 2)
                                      + ((rep_site['y_hist'] - -2000) ** 2))
        
# %% Plot
  
# Get lab
lab_number_map, institution_map, lab_colors = labs()
rep_site['institute'] = rep_site.lab.map(institution_map)
sns.set_palette(lab_colors)

# Add all labs seperately for plotting
rep_site_no_all = rep_site.copy()
rep_site_no_all.loc[rep_site_no_all.shape[0] + 1, 'institute'] = 'All'
rep_site_all = rep_site.copy()
rep_site_all['institute'] = 'All'
rep_site_all = rep_site.append(rep_site_all)
rep_site_only_all = pd.DataFrame(columns=rep_site.columns)
rep_site_only_all['institute'] = rep_site['institute']
rep_site_only_all = rep_site_only_all.append(rep_site_all.loc[rep_site_all['institute'] == 'All'])

vars = ['n_good', 'median_fr', 'targeting_error']
ylabels =['Number of units', 'Median firing rate (spks/s)', 'Targeting error (um)']
ylims = [[0, 350],[0, 10], [0, 1000]]
sns.set(style="ticks", context="paper", font_scale=2)
f, axs = plt.subplots(1, 3, figsize=(18, 6), dpi=100)
for ax, v, ylab, ylim in zip(axs, vars, ylabels, ylims):
    sns.swarmplot(y=v, x='institute', data=rep_site_no_all, hue='institute',
                  palette=lab_colors, ax=ax)
    axbox = sns.boxplot(y=v, x='institute', data=rep_site_only_all, color='white',
                        showfliers=False, ax=ax)
    ax.set(ylabel=ylab, ylim=ylim, xlabel='')
    # [tick.set_color(lab_colors[i]) for i, tick in enumerate(ax5.get_xticklabels()[:-1])]
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=60)
    axbox.artists[-1].set_edgecolor('black')
    for j in range(5 * (len(axbox.artists) - 1), 5 * len(axbox.artists)):
        axbox.lines[j].set_color('black')
    ax.get_legend().set_visible(False)

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(FIG_PATH, 'variability_over_labs'))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 09:33:06 2022
By: Guido Meijer
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from datetime import datetime
from figure3.figure3_load_data import load_dataframe
from reproducible_ephys_functions import filter_recordings, labs
from one.api import ONE
one = ONE()

# Load in data
df_ins = load_dataframe(df_name='ins')
df_filt = filter_recordings(df_ins, min_rec_lab=0, min_lab_region=0, min_neuron_region=0)
df_use = df_filt[df_filt['lab_include'] == True]

for i, eid in enumerate(np.unique(df_use['eid'])):

    # Get age of mouse at the time of the recording
    ses_details = one.get_details(eid)
    sub_details = one.alyx.rest('subjects', 'list', nickname=ses_details['subject'])
    birth_date = datetime.date(datetime.strptime(sub_details[0]['birth_date'], '%Y-%m-%d'))
    age = (ses_details['date'] - birth_date).days
    df_use.loc[df_use['eid'] == eid, 'age'] = age

# %% Plot

lab_number_map, institution_map, lab_colors = labs()
cmap = []
for i, inst in enumerate(df_use['institute'].unique()):
    cmap.append(lab_colors[inst])

f, axs = plt.subplots(1, 5, figsize=(14, 3), dpi=300, sharey=False)
for i, region in enumerate(np.unique(df_use['region'])):
    sns.regplot(x='age', y='lfp_power', data=df_use[df_use['region'] == region], ax=axs[i],
                ci=None, scatter_kws={'color': 'w'})
    sns.scatterplot(x='age', y='lfp_power', data=df_use[df_use['region'] == region], hue='institute',
                    ax=axs[i], legend=None, palette=cmap)
    r, p = pearsonr(df_use.loc[(df_use['region'] == region) & (~df_use['lfp_power'].isnull()), 'age'],
                    df_use.loc[(df_use['region'] == region) & (~df_use['lfp_power'].isnull()), 'lfp_power'])
    axs[i].set(title=f'{region}, p={p:.2f}', xlabel='Age (days)')
    if i == 0:
        axs[i].set(ylabel='LFP power (db)')
    else:
        axs[i].set(ylabel='')

plt.tight_layout()
sns.despine(trim=False)
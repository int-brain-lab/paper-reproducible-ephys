# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 18:28:17 2021

@author: Noam Roth

Variabity within/across labs

"""

#%% Imports and setup
from os.path import join, isdir
from os import mkdir
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from reproducible_ephys_functions import data_path, labs
from reproducible_ephys_paths import FIG_PATH


#%% Compute the variance across and within labs

def var_within_across_labs(metrics, metric):
    
    
    # compute a mean of that metric for each institution
    mean_per_lab = metrics.groupby('institution',as_index=False).mean()[metric]
    # compute variance across labs as the variance of these means
    var_across = np.nanvar(mean_per_lab)    
    # also add those means as a column in the df
    column_name = 'mean_' + str(metric)
    metrics[column_name] = metrics.groupby('institution')[metric].transform('mean')   
    #compute variance within labs by subtracting the lab specific mean from each recording, square, then average
    var_within = np.nanmean([(metrics[metric] - metrics[column_name]) ** 2])  

    return var_across/var_within      
    
#%%
# Load in data
metrics = pd.read_csv(join(data_path(), 'metrics_session.csv'))
lab_number_map, institution_map, lab_colors = labs()
metrics['institution'] = metrics.lab.map(institution_map)
# replace inf with nans for  groupby operation
metrics = metrics.replace([np.inf, -np.inf], np.nan)

#compute variance measure for all reasonable metrics
for i in range(6,15):
    metric = metrics.keys()[i]
    percent = var_within_across_labs(metrics,metric) * 100
    print('Variance (across/within) of %s : %s %%\n' %(metric,percent) )



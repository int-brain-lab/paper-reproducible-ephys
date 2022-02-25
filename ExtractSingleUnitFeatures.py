#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Marsa Taheri

"""

import numpy as np
import pickle
import pandas as pd
from reproducible_ephys_paths import FIG_PATH
#import matplotlib.pyplot as plt

data_FF_Rstim = pd.DataFrame()
REGIONS = ['PPC', 'CA1', 'DG', 'LP', 'PO']
#Loop through all brain regions
for k, region in enumerate(REGIONS):
    # Load the dataframes for each brain region
    data_perievent = pickle.load(open('data/SingleUnitData/' + region + '_PeriEventFRandFF_DF_SlidingWind_Pyk_restrTrials_010622.pkl', "rb"))
    data_avg = pickle.load(open('data/SingleUnitData/' + region + '_ClusterFeatures_DF_SlidingWind_Pyk_restrTrials_010622.pkl', "rb"))

    Subj_all =[]; FF_right_all = [];
    # For each cluster, extract relevant data; here 'Subj' and 'FF_R':
    for c in range(len(data_perievent)):
        Subj = data_perievent['Subject'][c] #extracts the subject name
        Subj_all.append(Subj)
    
        # Extract time vector corresponding to FF over time data; we want 40 ms to 200 ms post event
        TimeVect = data_perievent['TimeVect_FF'][c] 
        start = np.where(TimeVect==0.04)[0][0]
        stop = np.where(TimeVect==0.2)[0][0]
    
        # Sometimes when there are NaNs in the FF, there's an extra dimension in the data; fix it:
        if np.shape(data_perievent['FFoverT'][c])[0]==1:
            data_perievent['FFoverT'][c] = np.squeeze(data_perievent['FFoverT'][c])
    
        #The data_perievent['event_Titles'][c][index] extracts the event title, i.e., info on how the corresponding FRs and FFs 
        # are calculated. Here, we want aligned to movement, correct choices, full contrast, for R and L stim, 
        #which have indices 11 and 14, in order. Also, the FF is only useful if the cluster's avg. firing rate is >=1.
        if data_avg['AvgFR'][c]>=1:
            #Right stim:
            FFoverT_R = data_perievent['FFoverT'][c][11]
            FF_right = np.mean(FFoverT_R[start:stop]) # mean FF of 40-200 ms post Right movement
            FF_right_all.append(FF_right)
        else:
            FF_right_all.append(float('nan'))
    
            # #Left stim (if needed):
            # FFoverT_L = data_perievent['FFoverT'][c][14]    
            # FF_left = np.mean(FFoverT_L[start:stop]) # mean FF of 40-200 ms post Left movement

        # To visualize plot of FF over time:
        #plt.plot(TimeVect, FFoverT_R)
    
    data_to_save = np.array([np.repeat(region, len(data_perievent)), Subj_all, FF_right_all])
    data_to_save = np.transpose(data_to_save);
    df = pd.DataFrame(data_to_save, columns=['brain_region', 'subject','FF_Rstim'],
                      index=np.arange(3, 3+len(data_perievent))) #index=np.arange(0,len(clusterIDs)))
    # Put dataframes from each brain region into one:
    data_FF_Rstim = data_FF_Rstim.append(df)
    # Save dataframe
    data_FF_Rstim.to_pickle(FIG_PATH+'/' + 'data_FF_Rstim.pkl')
    #To load: output = pd.read_pickle(FIG_PATH+'/' + 'data_FF_Rstim.pkl')

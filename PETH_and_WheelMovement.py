#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 12:19:57 2024

@author: Sebastian and Marsa
"""

from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import numpy as np
import figrid as fg
from reproducible_ephys_functions import filter_recordings, BRAIN_REGIONS, labs, save_figure_path, figure_style, save_data_path
from fig_taskmodulation.fig_taskmodulation_load_data import load_data, load_dataframe, tests, filtering_criteria
from fig_taskmodulation.fig_taskmodulation_plot_functions import plot_raster_and_psth, plot_raster_and_psth_LvsR
import seaborn as sns
import pandas as pd
import pickle
from matplotlib.transforms import Bbox
import json
from permutation_test import permut_test, permut_dist
from matplotlib import cm
from matplotlib.colors import ListedColormap, to_rgb

from one.api import ONE
import brainbox.behavior.wheel as wh
from brainbox.behavior.dlc import get_dlc_everything
from brainbox.task.trials import get_event_aligned_raster, filter_trials
one = ONE()

lab_number_map, institution_map, lab_colors = labs()

df = load_dataframe()
for event in ['move']: #['stim', 'move']:
    data = load_data(event=event, norm='subtract', smoothing='sliding')
    df_filt = filter_recordings(df, **filtering_criteria)
    # print(df_filt.groupby(['region', 'institute']).agg({'permute_include': 'count'}))
    all_frs_l = data['all_frs_l']
    all_frs_r = data['all_frs_r']
    all_frs_l_std = data['all_frs_l_std']
    all_frs_r_std = data['all_frs_r_std']
    all_frs_l = data['all_frs_l'][df_filt['permute_include'] == 1]
    all_frs_r = data['all_frs_r'][df_filt['permute_include'] == 1]
    all_frs_l_std = data['all_frs_l_std'][df_filt['permute_include'] == 1]
    all_frs_r_std = data['all_frs_r_std'][df_filt['permute_include'] == 1]
    df_filt = df_filt[df_filt['permute_include'] == 1].reset_index()
    df_filt_reg = df_filt.groupby('region')
    reg = 'PO'
    df_reg = df_filt_reg.get_group(reg)
    df_reg_inst = df_reg.groupby('institute')
    
    WheelAndPETH = pd.DataFrame()
    #inst = 'UCLA'
    for inst in list(np.unique(df_reg['institute'])): #['UCLA', 'CCU']: # For all IBL institutes, use: list(institution_map.values()) 
        df_inst = df_reg_inst.get_group(inst)
        inst_idx = df_reg_inst.groups[inst]
        # Select L vs R side:
        frs_inst_r = all_frs_l[inst_idx, :]
        print(frs_inst_r.shape)
        # plt.plot(data['time'], frs_inst_r.T)
        # plt.ylabel("Baselined firing rate (sp/s)")
        # plt.xlabel("Time from {} onset (s)".format(event))
        # plt.close()
        
        # plt.plot(data['time'], np.mean(frs_inst_r, axis=0))
        # plt.ylabel("Baselined firing rate (sp/s)")
        # plt.xlabel("Time from {} onset (s)".format(event))
        # plt.ylim(-0.2, 2.5)
        # plt.title("Average activity in {}".format(inst))
        # plt.savefig("total {} {}".format(event, inst))
        # plt.close()
        
        # #split by those with FR more than the median of all neurons summed over time:
        # med = np.median(frs_inst_r.sum(1))
        # plt.plot(data['time'], np.mean(frs_inst_r[frs_inst_r.sum(1) < med], axis=0))
        # plt.plot(data['time'], np.mean(frs_inst_r[frs_inst_r.sum(1) > med], axis=0))
        # plt.ylabel("Baselined firing rate (sp/s)")
        # plt.xlabel("Time from {} onset (s)".format(event))
        # plt.ylim(-1, 5)
        # plt.title("Average activity - median split - in {}".format(inst))
        # plt.savefig("total med split {} {}".format(event, inst))
        # plt.close()
        
        # Avg trace for each subject of the specified institute:
        subs, counts = np.unique(df_inst.subject, return_counts=True)
        fig, axs = plt.subplots(2, 1, figsize=(5, 6), constrained_layout=True)
        for s, c in zip(subs, counts):
            axs[0].plot(data['time'], np.mean(frs_inst_r[df_inst.subject == s], axis=0), label=s + " (n={})".format(c))
            
            
            #Get eid and wheel data:
            eid = np.unique(df_inst.eid[df_inst.subject == s])
            pid = np.unique(df_inst.pid[df_inst.subject == s])
            if event == 'move':
                align_event='firstMovement_times'
            elif event == 'stim':
                align_event='stimOn_times'
            epoch = [-0.2, 0.7]
            tbin=0.05
            contrast=(1, 0.5, 0.25, 0.125, 0.0625) #non-zero contrasts
            order='trial num'
            sort='choice and side'
            tbin2=100 #Need to check this
            
            wheel = one.load_object(eid, 'wheel', collection='alf')
            trials = one.load_object(eid, 'trials')
            wheel_velocity = wh.velocity(wheel.timestamps, wheel.position)
            wheel_raster, wheel_t = get_event_aligned_raster(wheel.timestamps, trials[align_event],
                                                       values=wheel_velocity, tbin=tbin, epoch=epoch)  
            wheel_raster_sorted, wheel_psth = filter_trials(trials, wheel_raster, align_event, contrast, order, sort)
            psth_lines=[]
            include_trials = 'left_correct' #right side stim, correct trials only
            for ps in wheel_psth.keys():
                if ps == include_trials:
                    psth_lines.append(axs[1].plot(wheel_t, wheel_psth[ps]['vals']/tbin2))
                    axs[1].fill_between(wheel_t, wheel_psth[ps]['vals'] / tbin2 + wheel_psth[ps]['err'] / tbin2,
                                    wheel_psth[ps]['vals'] / tbin2 - wheel_psth[ps]['err'] / tbin2,
                                    alpha=0.3) #, **wheel_psth[ps]['linestyle'])

            
            # Get behavior & other info & save in dataframe:
            wheel_vel = wheel_psth['left_correct']['vals']/tbin2 
            #wheel_vel = wheel_psth[include_trials]['vals']/tbin2
            t_0 = np.where(wheel_t>0)[0][0] #first index of time>0
            t_end = np.where(wheel_t > 0.25)[0][0] #OR: np.where(wheel_t>(wheel_t[t_0] + 0.25))[0][0] # index of time>0.25 s from start
            
            behav_pass = np.unique(df_inst.behavior[df_inst.subject == s])
            wheel_max = max(wheel_vel[t_0:]) #over entire time course after event
            wheel_min = min(wheel_vel[t_0:]) #over entire time course after event
            wheel_AUC_short = np.cumsum(wheel_vel[t_0:t_end+1])[-1] #over ~0.25 s
            wheel_AUC_long = np.cumsum(wheel_vel[t_0:])[-1] #over entire time course after event
            wheel_med_short = np.median(wheel_vel[t_0:t_end+1]) #over ~0.25 s
            wheel_med_long = np.median(wheel_vel[t_0:]) #over entire time course after event
            
            # Neuron firing rate averaged over recording for specified brain region:
            neur_avgFR = np.mean(frs_inst_r[df_inst.subject == s], axis=0)
            t_0_neur = np.where(data['time']>0)[0][0] 
            t_end_neur = np.where(data['time'] > 0.25)[0][0] 
            neur_med = np.median(neur_avgFR[t_0_neur:t_end_neur+1]) #over ~0.25 s
            neur_avg = np.mean(neur_avgFR[t_0_neur:t_end_neur+1]) #over ~0.25 s
            
            # Individual neuron FRs:
            #neur_indiv_avg = np.mean(frs_inst_r[df_inst.subject == s][:,t_0_neur:t_end_neur+1],axis=1) #avg over time of each recorded neuron

            # Save into dataframe:
            data_Wheel_PETH = np.array([inst, s, eid, pid, behav_pass, include_trials,
                                        wheel_max, wheel_min, 
                                        wheel_AUC_short, wheel_AUC_long, wheel_med_short, 
                                        reg, neur_med, neur_avg], dtype=object)
            #data_Wheel_PETH = np.transpose(data_Wheel_PETH);
            df_toAppend = pd.DataFrame(data_Wheel_PETH.reshape(1,14), 
                                       columns=['instit', 'subj', 'eid', 'pid', 'behav_pass', 'include_trials',
                                                'wheel_max','wheel_min', 
                                                'wheel_AUC_short', 'wheel_AUC_long', 'wheel_med_short', 
                                                'brain_reg', 'neur_med', 'neur_avg'],
                                       index=np.arange(len(WheelAndPETH), len(WheelAndPETH)+1))
            
            WheelAndPETH = pd.concat([WheelAndPETH, df_toAppend], ignore_index=True) #new method: instead of 'append' use 'concat'
        
        
        ax=axs[0]
        ax.set_ylabel("Baselined firing rate (sp/s)")
        #ax.set_xlabel("Time from {} onset (s)".format(event))
        ax.set_ylim(-2, 10)
        ax.set_xlim(epoch[0], epoch[1])
        ax.legend()
        ax.vlines(0, *ax.get_ylim(), color='k', linestyle='dashed')
        ax.set_title("Average activity - subjects in {} ({})".format(inst, include_trials))

        ax=axs[1]
        ax.set_ylabel("Wheel Velocity (rad/s)")
        ax.set_ylim(-7.3, 0.88) #For Left side: ax.set_ylim(-7.3, 0.88), #For Right side: ax.set_ylim(-2, 3)
        ax.set_xlabel("Time from {} onset (s)".format(event))
        ax.set_xlim(epoch[0], epoch[1])
        ax.vlines(0, *ax.get_ylim(), color='k', linestyle='dashed')
        
        #plt.ylabel("Baselined firing rate (sp/s)")
        #plt.xlabel("Time from {} onset (s)".format(event))
        #plt.ylim(-2, 10)
        #plt.legend()
        #plt.title("Average activity - individuals - in {}".format(inst))
        #plt.savefig("indiv split {} {}".format(event, inst))
        ##plt.close()
        plt.show()
        # fig.savefig("indiv split {} {}".format(event, inst))
        
        
   #      # Avg trace for each subject of institute split into higher and lower than median:
   #      subs, counts = np.unique(df_inst.subject, return_counts=True)
   #      for s, c in zip(subs, counts):
   #          med = np.median(frs_inst_r[df_inst.subject == s].sum(1))
   #          p = plt.plot(data['time'], np.mean(frs_inst_r[np.logical_and(frs_inst_r.sum(1) > med, df_inst.subject == s)], axis=0), label=s)
   #          plt.plot(data['time'], np.mean(frs_inst_r[np.logical_and(frs_inst_r.sum(1) < med, df_inst.subject == s)], axis=0), '--', color=p[0].get_color())
   #      plt.ylabel("Baselined firing rate (sp/s)")
   #      plt.xlabel("Time from {} onset (s)".format(event))
   #      plt.ylim(-4, 16)
   #      plt.legend()
   #      plt.title("Average activity - individuals median split - in {}".format(inst))
   #      plt.savefig("indiv med split {} {}".format(event, inst))
   #      #plt.show()
   #      plt.close()
        
        
   #      # #Plot individual neurons:
   #      # subs, counts = np.unique(df_inst.subject, return_counts=True)
   #      # for s, c in zip(subs, counts):
   #      #     plt.plot(data['time'],frs_inst_r[df_inst.subject == s].T)
   #      #     plt.ylabel("Baselined firing rate (sp/s)")
   #      #     plt.xlabel("Time from {} onset (s)".format(event))
   #      #     #plt.xlim(epoch[0], epoch[1])
   #      #     plt.vlines(0, *plt.ylim(), color='k', linestyle='dashed')
   #      #     plt.title("Indiv. neuron activity {} in ({})".format(s, include_trials))
   #      #     plt.plot(data['time'], np.mean(frs_inst_r[df_inst.subject == s], axis=0), 'k', linewidth=1.2)
   #      #     plt.show()
   #      #     #fig.savefig("indiv neurons of subj {} in {} {}".format(s, inst, event))        


   # # Plot correlations:
   #      #fig2, axs2 = plt.subplots(2, 1, figsize=(5, 6), constrained_layout=True)
   #      #axs2[0].scatter(np.array(WheelAndPETH['wheel_max']), np.array(WheelAndPETH['neur_avg']), c=np.array(WheelAndPETH.index))#.format(c))
   #      #axs2[0].scatter(np.array(WheelAndPETH['wheel_max']), np.array(WheelAndPETH['neur_avg']), label=inst)
    
    Xvar = 'wheel_min' #'wheel_AUC_short' #
    Yvar = 'neur_avg'
    dataX = np.array(WheelAndPETH[Xvar])
    dataY = np.array(WheelAndPETH[Yvar])
    labs = WheelAndPETH['instit'].values
    plt.scatter(dataX, dataY, c = [lab_colors[x] for x in labs])#, c=lab_colors[WheelAndPETH['instit']])
    #plt.legend()
    plt.xlabel(Xvar) 
    plt.xlabel("Min wheel velocity over 0.7 s (rad/s)") #("Total displacement over ~0.25 s") #
    plt.ylabel(Yvar) 
    plt.ylabel("Avg FR of subj neurons over 0.25 s (sp/s)")
    corr, pvalue = pearsonr(dataX, dataY)
    print('Pearsons correlation: %.3f' % corr)
    print('Pearsons corr pval = %.3f' % pvalue)
    #print("Pearsons correlation: {} (p={})".format(corr, pvalue))
    #plt.title("Post {}, {}".format(event, include_trials))
    plt.title("{}: Post {}, left correct trials".format(reg, event))



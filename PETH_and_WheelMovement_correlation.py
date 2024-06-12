#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 23:08:37 2024

@author: mt

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
from matplotlib import cm, colors
from matplotlib.colors import ListedColormap, to_rgb

from one.api import ONE
import brainbox.behavior.wheel as wh
from brainbox.behavior.dlc import get_dlc_everything
from brainbox.task.trials import get_event_aligned_raster, filter_trials
from figure7.figure7_load_data import load_regions
one = ONE()

lab_number_map, institution_map, lab_colors = labs()
reg_cent_of_mass = load_regions()

############################################################
# Select contrast and one of the below to load the specific dataframe:

contrast=1 #(1, 0.5, 0.25, 0.125, 0.0625) #non-zero contrasts, or 1 or 0.0625
    
# #Option 1 (all contrasts):
# df = load_dataframe() #This loads the dataframe for all onre
# event = 'move' #or, 'stim'
# data = load_data(event=event, norm='subtract', smoothing='sliding')


#Option 2 (specified contrasts):
df_path1 = save_data_path(figure='fig_taskmodulation').joinpath('fig_taskmodulation_dataframe_Contrast100.csv')
#.joinpath('fig_taskmodulation_dataframe_Contrast6p25.csv')
df = pd.read_csv(df_path1)

event = 'move' #or, 'stim'
norm='subtract'
smoothing='sliding'
df_path2 = save_data_path(figure='fig_taskmodulation').joinpath(
#    f'fig_taskmodulation_data_event_{event}_smoothing_{smoothing}_norm_{norm}_Contrast6p25.npz')
    f'fig_taskmodulation_data_event_{event}_smoothing_{smoothing}_norm_{norm}_Contrast100.npz')
data = dict(np.load(df_path2, allow_pickle=True))
############################################################


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
cofm_reg = reg_cent_of_mass[reg_cent_of_mass['region'] == reg] #target xyz position for region
df_reg_inst = df_reg.groupby('institute')
 
WheelAndPETH = pd.DataFrame()
IndNeurons = pd.DataFrame()

for include_trials in ['right_correct', 'left_correct']: #right side stim, correct trials only
    #inst = 'UCLA'
    for inst in list(np.unique(df_reg['institute'])): #['UCLA', 'CCU', 'UCL', 'SWC', 'Princeton']: # list(np.unique(df_reg['institute'])): #['UCLA']: ##For all IBL institutes, use: list(institution_map.values()) 
        df_inst = df_reg_inst.get_group(inst)
        inst_idx = df_reg_inst.groups[inst]
        
        # Select L vs R side:
        if include_trials == 'right_correct':
            frs_instit = all_frs_r[inst_idx, :]
        elif include_trials == 'left_correct':
            frs_instit = all_frs_l[inst_idx, :]  
        #print(frs_instit.shape)
    
    
        #Cluser x,y:
        #x_inst = df_filt['x'][inst_idx] #df_filt['x'][inst_idx] is the same as df_inst['x']
        

        # Avg trace for each subject of the specified institute:
        subs, counts = np.unique(df_inst.subject, return_counts=True)
        fig, axs = plt.subplots(2, 1, figsize=(5, 6), constrained_layout=True)
        for s, c in zip(subs, counts):
            axs[0].plot(data['time'], np.mean(frs_instit[df_inst.subject == s], axis=0), label=s + " (n={})".format(c))
             
            #Get eid and wheel data:
            eid = np.unique(df_inst.eid[df_inst.subject == s])
            pid = np.unique(df_inst.pid[df_inst.subject == s])
            if event == 'move':
                align_event='firstMovement_times'
            elif event == 'stim':
                align_event='stimOn_times'
            epoch = [-0.2, 0.7]
            tbin=0.05
            #contrast=(1, 0.5, 0.25, 0.125, 0.0625) #non-zero contrasts
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
            for ps in wheel_psth.keys():
                if ps == include_trials:
                    psth_lines.append(axs[1].plot(wheel_t, wheel_psth[ps]['vals']/tbin2))
                    axs[1].fill_between(wheel_t, wheel_psth[ps]['vals'] / tbin2 + wheel_psth[ps]['err'] / tbin2,
                                    wheel_psth[ps]['vals'] / tbin2 - wheel_psth[ps]['err'] / tbin2,
                                    alpha=0.3) #, **wheel_psth[ps]['linestyle'])
    
             
            # Get behavior & other info & save in dataframe:
            #wheel_vel = wheel_psth['right_correct']['vals']/tbin2 
            wheel_vel = wheel_psth[include_trials]['vals']/tbin2
            t_0 = np.where(wheel_t>0)[0][0] #first index of time>0
            t_end = np.where(wheel_t > 0.25)[0][0] #OR: np.where(wheel_t>(wheel_t[t_0] + 0.25))[0][0] # index of time>0.25 s from start
             
            behav_pass = np.unique(df_inst.behavior[df_inst.subject == s])
            wheel_max = max(wheel_vel[t_0:]) #over entire time course after event
            wheel_min = min(wheel_vel[t_0:]) #over entire time course after event
            wheel_abs_max = max(abs(wheel_vel[t_0:]))
            wheel_AUC_short = np.cumsum(wheel_vel[t_0:t_end+1])[-1] #over ~0.25 s
            wheel_AUC_long = np.cumsum(wheel_vel[t_0:])[-1] #over entire time course after event
            wheel_med_short = np.median(wheel_vel[t_0:t_end+1]) #over ~0.25 s
            wheel_med_long = np.median(wheel_vel[t_0:]) #over entire time course after event
             
            # Neuron firing rate averaged over recording for specified brain region:
            neur_avgFR = np.mean(frs_instit[df_inst.subject == s], axis=0)
            t_0_neur = np.where(data['time']>0)[0][0] 
            t_end_neur = np.where(data['time'] > 0.25)[0][0] 
            neur_med = np.median(neur_avgFR[t_0_neur:t_end_neur+1]) #over ~0.25 s
            neur_avg = np.mean(neur_avgFR[t_0_neur:t_end_neur+1]) #over ~0.25 s
            neur_change = np.mean(neur_avgFR[t_0_neur:t_end_neur+1]) - np.mean(neur_avgFR[0:t_0_neur]) #FR over ~0.25 s - pre-move FR
            neur_max = np.max(neur_avgFR[t_0_neur:t_end_neur+1]) #over ~0.25 s
            #neur_AbsMax = np.max(np.abs(neur_avgFR[t_0_neur:t_end_neur+1])) #over ~0.25 s
            
            # Individual neuron FRs:
            neur_indiv_avg = np.mean(frs_instit[df_inst.subject == s][:,t_0_neur:t_end_neur+1],axis=1) #avg over time of each recorded neuron
            neur_indiv_max = np.max(frs_instit[df_inst.subject == s][:,t_0_neur:t_end_neur+1],axis=1) #avg over time of each recorded neuron
            #neur_indiv_AbsMax = np.max(np.abs(frs_instit[df_inst.subject == s][:,t_0_neur:t_end_neur+1],axis=1)) #avg over time of each recorded neuron
            neur_indiv_change = np.mean(frs_instit[df_inst.subject == s][:,t_0_neur:t_end_neur+1],axis=1) - np.mean(frs_instit[df_inst.subject == s][:,0:t_0_neur],axis=1)
            
            #xyz positions of all units of the subject:
            x_pos = df_filt['x'][inst_idx][df_inst.subject == s]
            y_pos = df_filt['y'][inst_idx][df_inst.subject == s]
            z_pos = df_filt['z'][inst_idx][df_inst.subject == s]
            clustID = df_filt['cluster_ids'][inst_idx][df_inst.subject == s]
    
    
            # Save into dataframe:
            data_Wheel_PETH = np.array([inst, s, eid, pid, behav_pass, include_trials,
                                        wheel_max, wheel_min, wheel_abs_max,
                                        wheel_AUC_short, wheel_AUC_long, wheel_med_short, 
                                        reg, neur_med, neur_avg, neur_change, neur_max,
                                        np.mean(x_pos), np.mean(y_pos), np.mean(z_pos)], dtype=object)
            #data_Wheel_PETH = np.transpose(data_Wheel_PETH);
            df_toAppend = pd.DataFrame(data_Wheel_PETH.reshape(1,len(data_Wheel_PETH)),
                                       columns=['instit', 'subj', 'eid', 'pid', 'behav_pass', 'include_trials',
                                                'wheel_max','wheel_min', 'wheel_abs_max',
                                                'wheel_AUC_short', 'wheel_AUC_long', 'wheel_med_short', 
                                                'brain_reg', 'neur_med', 'neur_avg', 'neur_change', 'neur_max',
                                                'meanX','meanY','meanZ'],
                                       index=np.arange(len(WheelAndPETH), len(WheelAndPETH)+1))
             
            WheelAndPETH = pd.concat([WheelAndPETH, df_toAppend], ignore_index=True) #new method: instead of 'append' use 'concat'
         
            
             
            # Save into 2nd dataframe which includes info for individual neurons:
            data_indiv = np.array([np.repeat(inst, len(x_pos)), np.repeat(s, len(x_pos)), 
                                   np.repeat(eid, len(x_pos)), np.repeat(pid, len(x_pos)), 
                                   clustID, np.repeat(reg, len(x_pos)),
                                   np.repeat(behav_pass, len(x_pos)), np.repeat(include_trials, len(x_pos)),
                                   np.repeat(neur_med, len(x_pos)), np.repeat(neur_avg, len(x_pos)),
                                   np.repeat(neur_change, len(x_pos)), np.repeat(neur_max, len(x_pos)),
                                   neur_indiv_avg, neur_indiv_max, neur_indiv_change,
                                   x_pos, y_pos, z_pos], dtype=object)
            data_indiv2=data_indiv.T #data_indiv2 = np.transpose(data_indiv)
            df_indiv_toAppend = pd.DataFrame(data_indiv2,
                                             columns=['instit', 'subj', 'eid', 'pid', 
                                                      'cluster_ids', 'brain_reg',
                                                      'behav_pass', 'include_trials',
                                                      'neur_med', 'neur_avg', 'neur_change', 'neur_max',
                                                      'neur_indiv_avg', 'neur_indiv_max','neur_indiv_change',
                                                      'x_pos','y_pos','z_pos'],
                                             index=np.arange(len(IndNeurons), len(IndNeurons)+len(x_pos)))
            IndNeurons = pd.concat([IndNeurons, df_indiv_toAppend], ignore_index=True)
            
         
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
        if include_trials == 'right_correct':
            ax.set_ylim(-2, 3)#6)
        elif include_trials == 'left_correct':
            ax.set_ylim(-7.3, 0.88) #For Left side: ax.set_ylim(-7.3, 0.88), #For Right side: 
        ax.set_xlabel("Time from {} onset (s)".format(event))
        ax.set_xlim(epoch[0], epoch[1])
        ax.vlines(0, *ax.get_ylim(), color='k', linestyle='dashed')
         
        plt.show()
        fig.savefig("indiv split {} {} ({})".format(event, inst, include_trials))
         
    
    # # Plot correlations:
    #      #fig2, axs2 = plt.subplots(2, 1, figsize=(5, 6), constrained_layout=True)
    #      #axs2[0].scatter(np.array(WheelAndPETH['wheel_max']), np.array(WheelAndPETH['neur_avg']), c=np.array(WheelAndPETH.index))#.format(c))
    #      #axs2[0].scatter(np.array(WheelAndPETH['wheel_max']), np.array(WheelAndPETH['neur_avg']), label=inst)
     
    Xvar = 'wheel_AUC_short' #'wheel_abs_max' #'wheel_max' #'wheel_AUC_short' #
    Yvar = 'neur_change' #'neur_avg'
    dataX = np.array(WheelAndPETH[Xvar][WheelAndPETH['include_trials']==include_trials])
    dataY = np.array(WheelAndPETH[Yvar][WheelAndPETH['include_trials']==include_trials])
    labs = WheelAndPETH['instit'][WheelAndPETH['include_trials']==include_trials].values
    plt.scatter(dataX, dataY, c = [lab_colors[x] for x in labs])#, c=lab_colors[WheelAndPETH['instit']])
    #plt.legend()
    plt.xlabel(Xvar) 
    #plt.xlabel("Max |wheel velocity| over 0.7 s (rad/s)") #("Total displacement over ~0.25 s") #
    plt.ylabel(Yvar) 
    #plt.ylabel("Avg FR of subj neurons over 0.25 s (sp/s)")
    corr, pvalue = pearsonr(dataX, dataY)
    print('Pearsons correlation: %.3f' % corr)
    print('Pearsons corr pval = %.3f' % pvalue)
    #print("Pearsons correlation: {} (p={})".format(corr, pvalue))
    #plt.title("Post {}, {}".format(event, include_trials))
    plt.title("{}: Post {}, {} trials (contrast = {})".format(reg, event, include_trials, contrast))
    plt.show()
    

Xvar = 'wheel_AUC_short' #'wheel_med_short' #'wheel_max' #'wheel_AUC_short' #
Yvar = 'neur_change' #'neur_avg'
dataX = np.array(WheelAndPETH[Xvar])#[WheelAndPETH['behav_pass']==True]) #include left and right datapoints 
dataY = np.array(WheelAndPETH[Yvar])#[WheelAndPETH['behav_pass']==True]) #include left and right datapoints
labs = WheelAndPETH['instit'].values#[WheelAndPETH['behav_pass']==True].values
plt.scatter(dataX, dataY, c = [lab_colors[x] for x in labs])#, c=lab_colors[WheelAndPETH['instit']])
#plt.legend()
plt.xlabel(Xvar) 
#plt.xlabel("Max |wheel velocity| over 0.7 s (rad/s)") #("Total displacement over ~0.25 s") #
plt.ylabel(Yvar) 
#plt.ylabel("Avg FR of subj neurons over 0.25 s (sp/s)")
corr, pvalue = pearsonr(dataX, dataY)
print('Pearsons correlation: %.3f' % corr)
print('Pearsons corr pval = %.3f' % pvalue)
#print("Pearsons correlation: {} (p={})".format(corr, pvalue))
#plt.title("Post {}, {}".format(event, include_trials))
plt.title("{}: Post {}, R & L correct trials (contrast = {})".format(reg, event, contrast))
#plt.vlines(0, plt.ylim(), color='k', linestyle='dashed')
#plt.hlines(plt.xlim(), 0, color='k', linestyle='dashed')
plt.show()



side_select = 'right_correct'
Xvar = 'wheel_AUC_short' #'wheel_med_short' #'wheel_max' #'wheel_AUC_short' #
Yvar = 'neur_change' #'neur_avg'
dataX = np.array(WheelAndPETH[Xvar][WheelAndPETH['include_trials']==side_select]) #include left and right datapoints 
dataY = np.array(WheelAndPETH[Yvar][WheelAndPETH['include_trials']==side_select]) #include left and right datapoints
labs = WheelAndPETH['instit'].values[WheelAndPETH['include_trials']==side_select]
plt.scatter(dataX, dataY, c = [lab_colors[x] for x in labs])#, c=lab_colors[WheelAndPETH['instit']])
#plt.legend()
plt.xlabel(Xvar) 
#plt.xlabel("Max |wheel velocity| over 0.7 s (rad/s)") #("Total displacement over ~0.25 s") #
plt.ylabel(Yvar) 
#plt.ylabel("Avg FR of subj neurons over 0.25 s (sp/s)")
corr, pvalue = pearsonr(dataX, dataY)
print('Pearsons correlation: %.3f' % corr)
print('Pearsons corr pval = %.3f' % pvalue)
plt.title("{}: Post {}, {} trials (contrast = {})".format(reg, event, side_select, contrast))
plt.show()


# Correlation of Average X Y Z position with Neural response:
side_select = 'right_correct'
Xvar = 'meanY' #'wheel_med_short' #'wheel_max' #'wheel_AUC_short' #
Yvar = 'neur_change' #'neur_avg'
dataX = np.array(WheelAndPETH[Xvar][WheelAndPETH['include_trials']==side_select]) #include left and right datapoints 
dataY = np.array(WheelAndPETH[Yvar][WheelAndPETH['include_trials']==side_select]) #include left and right datapoints
labs = WheelAndPETH['instit'].values[WheelAndPETH['include_trials']==side_select]
plt.scatter(dataX, dataY, c = [lab_colors[x] for x in labs])#, c=lab_colors[WheelAndPETH['instit']])
#plt.legend()
plt.xlabel(Xvar) 
#plt.xlabel("Max |wheel velocity| over 0.7 s (rad/s)") #("Total displacement over ~0.25 s") #
plt.ylabel(Yvar) 
#plt.ylabel("Avg FR of subj neurons over 0.25 s (sp/s)")
corr, pvalue = pearsonr(dataX, dataY)
print('Pearsons correlation: %.3f' % corr)
print('Pearsons corr pval = %.3f' % pvalue)
plt.title("{}: Post {}, {} trials (contrast = {})".format(reg, event, side_select, contrast))
plt.show()



# Correlation of Individual X Y Z position with Averaged Neural response:
fig = plt.figure(figsize=(7,5))
side_select = 'right_correct'
Xvar = 'y_pos' 
Yvar = 'neur_change' #'neur_avg'
dataX = np.array(IndNeurons[Xvar][IndNeurons['include_trials']==side_select]) #include left and right datapoints 
dataY = np.array(IndNeurons[Yvar][IndNeurons['include_trials']==side_select]) #include left and right datapoints
labs = IndNeurons['instit'].values[IndNeurons['include_trials']==side_select]
plt.scatter((dataX - cofm_reg['y'].values) * 1e6, dataY, s=2, 
            c = [lab_colors[x] for x in labs])#, c=lab_colors[WheelAndPETH['instit']])
#plt.legend()
plt.xlabel(r'$\Delta$Y (A-P)') #(r'$\Delta$X (M-L)') #(Xvar) 
#plt.xlabel("Max |wheel velocity| over 0.7 s (rad/s)") #("Total displacement over ~0.25 s") #
plt.ylabel('Change in FR (averaged across subject neurons)')#(Yvar) 
#plt.ylabel("Avg FR of subj neurons over 0.25 s (sp/s)")
corr, pvalue = pearsonr(dataX, dataY)
print('Pearsons correlation: %.3f' % corr)
print('Pearsons corr pval = %.3f' % pvalue)
plt.title("{}: Post {}, {} trials (contrast = {})".format(reg, event, side_select, contrast))
# FOR A LEGEND:
# Create a dictionary of only the lab colors used in plot, then generate custom fake lines that will be used as legend entries:
keys = np.unique(labs)
dict2 = {x:lab_colors[x] for x in keys}
markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', markersize = '3', linestyle='') for color in dict2.values()]
plt.legend(markers, dict2.keys(), numpoints=1, fontsize=8)
plt.show()



#Plot for individual neurons:
Xvar = 'y_pos' 
Yvar = 'neur_indiv_change' #'neur_indiv_avg' #'neur_avg'
dataX = np.array(IndNeurons[Xvar])#[WheelAndPETH['behav_pass']==True]) #include left and right datapoints 
dataY = np.array(IndNeurons[Yvar])#[WheelAndPETH['behav_pass']==True]) #include left and right datapoints
labs = IndNeurons['instit'].values#[WheelAndPETH['behav_pass']==True].values
plt.scatter((dataX- cofm_reg['y'].values) * 1e6, dataY, s=1, c = [lab_colors[x] for x in labs])#, c=lab_colors[WheelAndPETH['instit']])
#plt.legend()
plt.xlabel(Xvar) 
#plt.xlabel("Max |wheel velocity| over 0.7 s (rad/s)") #("Total displacement over ~0.25 s") #
plt.ylabel(Yvar) 
#plt.ylabel("Avg FR of subj neurons over 0.25 s (sp/s)")
corr, pvalue = pearsonr(dataX, dataY)
print('Pearsons correlation: %.3f' % corr)
print('Pearsons corr pval = %.3f' % pvalue)
#print("Pearsons correlation: {} (p={})".format(corr, pvalue))
#plt.title("Post {}, {}".format(event, include_trials))
plt.title("{}: Post {}, {} trials (contrast = {})".format(reg, event, include_trials, contrast))
#plt.vlines(0, plt.ylim(), color='k', linestyle='dashed')
#plt.hlines(plt.xlim(), 0, color='k', linestyle='dashed')
plt.show()



#2D positional plot with colors showing lab ID:
fig = plt.figure(figsize=(7,7))
Xvar = 'x_pos' 
Yvar = 'y_pos'
dataX = np.array(IndNeurons[Xvar]) #[WheelAndPETH['behav_pass']==True]) #include left and right datapoints 
dataY = np.array(IndNeurons[Yvar])#[WheelAndPETH['behav_pass']==True]) #include left and right datapoints
labs = IndNeurons['instit'].values#[WheelAndPETH['behav_pass']==True].values
plt.scatter((dataX - cofm_reg['x'].values) * 1e6, (dataY - cofm_reg['y'].values) * 1e6, 
            s=2, c = [lab_colors[x] for x in labs])#, c=lab_colors[WheelAndPETH['instit']])
plt.xlabel(r'$\Delta$X (M-L)')#(Xvar) 
plt.ylabel(r'$\Delta$Y (A-P)')#(Yvar) 
plt.title("Position of neurons in {}".format(reg))
# FOR A LEGEND:
# Create a dictionary of only the lab colors used in plot, then generate custom fake lines that will be used as legend entries:
keys = np.unique(labs)
dict2 = {x:lab_colors[x] for x in keys}
markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', markersize = '3', linestyle='') for color in dict2.values()]
plt.legend(markers, dict2.keys(), numpoints=1, fontsize=10)
plt.gca().set_aspect('equal')
plt.show()



#2D positional plot with colors showing neural firing info (trial averaged and often averaged over time):
fig = plt.figure(figsize=(7,7))
Xvar = 'x_pos' 
Yvar = 'y_pos'
ColorVar = 'neur_change' #'neur_max' #'neur_change' #'neur_avg'# 'neur_indiv_max' #'neur_indiv_avg' #'neur_avg'
dataX = np.array(IndNeurons[Xvar])#[IndNeurons['include_trials']==side_select])#[WheelAndPETH['behav_pass']==True]) #include left and right datapoints 
dataY = np.array(IndNeurons[Yvar])#[IndNeurons['include_trials']==side_select])#[WheelAndPETH['behav_pass']==True]) #include left and right datapoints
datacolor = np.array(IndNeurons[ColorVar])#[IndNeurons['include_trials']==side_select].astype(float))
datacolor = abs(datacolor) #absolute value of change in neuronal activity
#datacolor = np.log10(abs(datacolor)) #log of absolute value of change in neuronal activity

#norm = colors.Normalize(vmin=min(datacolor), vmax=max(datacolor), clip=False)#(vmin=10, vmax=max(datacolor), clip=False)
# mapper = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap('viridis'))
# cluster_color = np.array([mapper.to_rgba(col) for col in IndNeurons[ColorVar]])

plt.scatter((dataX - cofm_reg['x'].values) * 1e6, (dataY - cofm_reg['y'].values) * 1e6, 
            s=2, c=datacolor, cmap='viridis') #c=cluster_color
plt.xlabel(r'$\Delta$X (M-L)')#(Xvar) 
plt.ylabel(r'$\Delta$Y (A-P)')#(Yvar) 
plt.colorbar(label= r'$\Delta$FR (averaged across subj neurons)')#= 'Max FR of neuronal average (sp/s)')#label=ColorVar)
#cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax)
#cbar.set_label('Fano Factor')
plt.title("{} neurons, {} trials (contrast = {})".format(reg, include_trials, contrast))

#cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax)
#cbar.set_label('Fano Factor')
plt.gca().set_aspect('equal')
plt.show()





# -*- coding: utf-8 -*-
"""
@author: Marsa Taheri
"""
import numpy as np
import pandas as pd
import seaborn as sns
from os.path import join
import matplotlib.pyplot as plt
import brainbox.io.one as bbone
from reproducible_ephys_functions import query, labs, eid_list, combine_regions
from reproducible_ephys_paths import FIG_PATH
from one.api import ONE
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.tri as mtri

from pylab import *
import scipy.io as sio       
from brainbox.singlecell import calculate_peths
from brainbox.plot import peri_event_time_histogram, firing_rate
from brainbox.metrics.single_units import spike_sorting_metrics, quick_unit_metrics
import pickle
  
from ibllib.atlas import atlas, AllenAtlas
from ibllib.pipes import histology
from iblutil.util import Bunch

one = ONE()
ba = AllenAtlas()

# Query repeated site trajectories
#traj = query()
#traj = query(behavior=True)
ListOfEIDs = eid_list()

# Initialize dataframe
ClusterFeatures = pd.DataFrame() #['eID', 'Lab ID','clusterID', 'AvgFR', 'AvgFF', 'amps', 'peak-trough', 'Xloc', 'Yloc', 'Zloc']
PeriEventFRandFF = pd.DataFrame() #['FR stimOn', 'FR '] FR and FF averaged over entire session

regions = 'PPC' #[PPC', 'CA1', 'DG', 'LP', 'PO']
binSzPeri = 0.1 #bin size for calculating perievent FR and FF
pre_time, post_time = 0.4, 0.8 
CapPostTime = 0.4 #The time point at which to cap the FR/FF post-event mean, so even though over time we have 0.8 s post-event, we can cap the analysis of mean post-event to 0.4 s 

SaveFigs=0 # To choose whether to save the figures & .pkl & .mat files at the end; to save, set equal to 1
SaveStr = '_Oct2021' #String to have at the end of the figure/file names to be saved, e.g., the date

include_NonRS = 0 #Whether or not to include non-RS that pass the area; 0 for No, 1 for Yes.


# %% Functions (modified from others' codes)
# used if firing rates are normalized
def normalise_act(activities, spikes, probe, clusterIDs, base_line_times):
    activity_pre, _ = calculate_peths(spikes[probe]['times'], spikes[probe]['clusters'],
                                      np.array(clusterIDs), base_line_times,
                                      pre_time=0.4, post_time=-0.2, smoothing=0, bin_size=0.01) #-0.2 s means 200 ms BEFORE the stim.
    baseline = np.mean(activity_pre.means, axis=1)

    normed_activities = []
    for a in activities:
        normed_activities.append(((a.means.T - baseline) / (1 + baseline)).T)

    return normed_activities


# retreives the trajectory for a list of eids
def get_traj(ListOfEIDs):
    traj = query()
    tmp = []
    for t in traj:
        if t['session']['id'] in ListOfEIDs:
            tmp.append(t)
    traj = tmp
    
    return traj


# %% Loop through repeated site recordings and extract the data
trajRS=get_traj(ListOfEIDs)
N_RSprobes = len(trajRS) #the rest in 'traj' are probes that passed the region but were not repeated site probes

#This line tries to run the code for the non-RS probes as well, unless they were not included (i.e., ListOfEIDs_updated not defined):
if include_NonRS:
    open_file = open('Close_NotRS_Traj', "rb")
    trajNotRS = pickle.load(open_file)
    open_file.close()
    traj = trajRS + trajNotRS #merge/concatenate the two lists
else:
    #traj=get_traj(ListOfEIDs)
    traj = trajRS


PykiloYes, PykiloNo, NotWorking, NoGoodClustersInRegion = [],[],[],[]
for i in range(len(traj)):
    print('Processing repeated site recording %d of %d' % (i+1, len(traj)))
    fr, FFofFR, LabNumArray, SpikeSortMethod = [], [], [], []
    xvals, yvals, zvals =[], [], []
    FRpre_AllEv, FFpre_AllEv, FRpost_AllEv, FFpost_AllEv =[], [], [], []
    event_TitlesAll = []
    TimeVect, FRoverT_AllEv, FFoverT_AllEv = [],[],[]
    Ntrials=[]

    # Load in data
    eid = traj[i]['session']['id']
    probe = traj[i]['probe_name']
    DateInfo = traj[i]['datetime']
    subj = traj[i]['session']['subject'] 

    #Is it a planned repeated site probe or not (e.g., nearby probe):
    if i<N_RSprobes:
        RS_YorN = 1
    else:
        RS_YorN = 0
        
    try:
        #Try Pykilosort: 
        spikes = one.load_object(eid, 'spikes', collection='alf/{}/pykilosort'.format(probe))
        clusters = one.load_object(eid, 'clusters', collection='alf/{}/pykilosort'.format(probe))
        channels = bbone.load_channel_locations(eid, one=one, probe=probe, aligned=True)
        clusters = bbone.merge_clusters_channels(dic_clus={probe: clusters}, channels=channels)[probe]
        
        BrainRegionsInProbe = combine_regions(clusters['acronym'])
        
        try:
            # Get relevant cluster ids
            clusterIDs  = clusters['metrics']['cluster_id'][BrainRegionsInProbe == regions][clusters['metrics']['label'] == 1]
        except:
            #If it doesn't have a 'metrics' key, then create one and remake the clusters object:
            #c, _ = spike_sorting_metrics(spikes.times, spikes.clusters, spikes.amps, spikes.depths, cluster_ids=np.arange(clusters.channels.size))
            #c, drift = spike_sorting_metrics(spikes.times, spikes.clusters, spikes.amps, spikes.depths)
            r = quick_unit_metrics(spikes.clusters, spikes.times, spikes.amps, spikes.depths, cluster_ids=np.arange(clusters.channels.size)) #Need to convert to a dataframe
            dict_metrics = {} #converting r (a util.Bunch object that's really a dictionary) to a regular dictionary
            for key in r.keys():
                dict_metrics[key] = r[key]
            df_metrics = pd.DataFrame.from_dict(dict_metrics) #converting dictionary to a Pandas dataframe
            clusters['metrics'] = df_metrics 
            clusterIDs  = clusters['metrics']['cluster_id'][BrainRegionsInProbe == regions][clusters['metrics']['label'] == 1]

        #? clusterIDs  = clusters['metrics']['cluster_id'][BrainRegionsInProbe == regions][clusters['metrics']['label'] == 1]
        PykiloYes.append(traj[i]['session']['subject'])
        SpikeSorter = 'Pykilosort'

    except:
        try:
            #Try ks2 instead of Pykilosort:
            spikes = one.load_object(eid, 'spikes', collection='alf/{}'.format(probe), revision='') #, revision='' needed or not?? otherwise uses latest revision
            clusters = one.load_object(eid, 'clusters', collection='alf/{}'.format(probe), revision='')
            channels = bbone.load_channel_locations(eid, one=one, probe=probe, aligned=True)
            clusters = bbone.merge_clusters_channels(dic_clus={probe: clusters}, channels=channels)[probe]
            
            BrainRegionsInProbe = combine_regions(clusters['acronym'])
            try:
                 #Get relevant cluster ids
                clusterIDs  = clusters['metrics']['cluster_id'][BrainRegionsInProbe == regions][clusters['metrics']['label'] == 1]
            except:
                r = quick_unit_metrics(spikes.clusters, spikes.times, spikes.amps, spikes.depths, cluster_ids=np.arange(clusters.channels.size)) #Need to convert to a dataframe
                dict_metrics = {} #converting r (a util.Bunch object that's really a dictionary) to a regular dictionary
                for key in r.keys():
                    dict_metrics[key] = r[key]
                df_metrics = pd.DataFrame.from_dict(dict_metrics) #cnverting dictionary to a Pandas dataframe
                ##c, drift = spike_sorting_metrics(spikes.times, spikes.clusters, spikes.amps, spikes.depths)
                clusters['metrics'] = df_metrics #before: = c
                clusterIDs  = clusters['metrics']['cluster_id'][BrainRegionsInProbe == regions][clusters['metrics']['label'] == 1]
            
            PykiloNo.append(traj[i]['session']['subject'])
            SpikeSorter = 'KS2'

        except:
            NotWorking.append(traj[i]['session']['subject'])
            print(traj[i]['session']['subject'], ' !!No Clusters!!')
            continue

    #May not need this part below (?):
    # Get coordinates of micro-manipulator and histology
    hist = one.alyx.rest('trajectories', 'list', provenance='Histology track',
                     probe_insertion=traj[i]['probe_insertion'])
    manipulator = one.alyx.rest('trajectories', 'list', provenance='Micro-manipulator',
                                probe_insertion=traj[i]['probe_insertion'])
    if (len(hist) == 0) or (len(manipulator) == 0):
        continue


    # Find lab name, and associate the lab name with a number:
    LabName = traj[i]['session']['lab']
    lab_number_map, _, _ = labs() #lab_number_map, institution_map, lab_colors = labs()
    LabNum = int(lab_number_map.get(LabName)[-1])
    

    # Get relevant cluster ids
    clusterIDs  = clusters['metrics']['cluster_id'][BrainRegionsInProbe == regions][clusters['metrics']['label'] == 1]
   # mask = np.logical_and(clusters['acronym'] == 'LP', clusters['metrics']['label'] == 1)
    if len(clusterIDs)==0: # when no good clusters are found in the brain region
        NoGoodClustersInRegion.append(traj[i]['session']['subject'])
        continue
    
    #Find cluster waveform (spike amp & width):
    amps1 = clusters['amps']
    ptt1 = clusters['peakToTrough']


    #Find specific task Event times:
    #event == 'Stim' 
    timesStimOn_orig = one.load_dataset(eid, '_ibl_trials.stimOn_times.npy') #one.load(eid, dataset_types=['trials.stimOn_times'])[0]
    base_line_times = one.load_dataset(eid, '_ibl_trials.stimOn_times.npy')[0] #one.load(eid, dataset_types=['trials.stimOn_times'])[0]
    base_line_times = base_line_times[~np.isnan(base_line_times)]
    contrast_L_orig =  one.load_dataset(eid, '_ibl_trials.contrastLeft.npy')
    contrast_R_orig = one.load_dataset(eid, '_ibl_trials.contrastRight.npy')
    contrast_L, contrast_R = contrast_L_orig[~np.isnan(timesStimOn_orig)], contrast_R_orig[~np.isnan(timesStimOn_orig)]
    timesStimOn = timesStimOn_orig[~np.isnan(timesStimOn_orig)]
    event_times_left = timesStimOn[contrast_L > 0]
    event_times_right = timesStimOn[contrast_R > 0]
    event_times_0 = timesStimOn[np.logical_or(contrast_R == 0, contrast_L == 0)] #'or' is used because the other contrast is always nan
    event_times_left100 = timesStimOn[contrast_L == 1]
    event_times_right100 = timesStimOn[contrast_R == 1]
    
    #event == 'Move'
    times1stMove_orig = one.load_dataset(eid, '_ibl_trials.firstMovement_times.npy') #one.load(eid, dataset_types=['trials.firstMovement_times'])[0]
    choice_orig = one.load_dataset(eid, '_ibl_trials.choice.npy') #one.load(eid, dataset_types=['trials.choice'])[0]
    if (~np.isnan(times1stMove_orig)).sum() < 300: # don't analyze if mouse made less than 300 choices/movement (even though >400 trials were done)
        continue
    choice = choice_orig[~np.isnan(times1stMove_orig)]
    times1stMove = times1stMove_orig[~np.isnan(times1stMove_orig)]
    event_times_Rchoice = times1stMove[choice == -1] # -1 means mouse reports stim on Right, so counter-clockwise wheel turn
    event_times_Lchoice = times1stMove[choice == 1] # 1 means mouse reports stim on Left, so clockwise wheel turn
    StimWithChoice = np.logical_and(~np.isnan(timesStimOn_orig), ~np.isnan(times1stMove_orig)) #all the indices where there was both a stim and a choice time
    #MoveandRstim = np.logical_and(contrast_R_orig>0, StimWithChoice) #events when stim was R (contrast>0) and there was a choice ultimately
    MoveandR100stim = np.logical_and(contrast_R_orig == 1, StimWithChoice) #events when stim was R (contrast>0) and there was a choice ultimately
    MoveandL100stim = np.logical_and(contrast_L_orig == 1, StimWithChoice) 
    eventMove_stR100chR = times1stMove_orig[np.logical_and(MoveandR100stim, choice_orig == -1)] #times of movement when choice was R and stim was R (contrast>0)
    eventMove_stR100chL = times1stMove_orig[np.logical_and(MoveandR100stim, choice_orig == 1)] #times of movement when choice was L and stim was R (contrast>0), so incorrect
    eventMove_stL100chR = times1stMove_orig[np.logical_and(MoveandL100stim, choice_orig == -1)]
    eventMove_stL100chL = times1stMove_orig[np.logical_and(MoveandL100stim, choice_orig == 1)] 
    eventStim_stR100chR = timesStimOn_orig[np.logical_and(MoveandR100stim, choice_orig == -1)] #times of stim onset when choice was R and stim was R (contrast>0)
    eventStim_stR100chL = timesStimOn_orig[np.logical_and(MoveandR100stim, choice_orig == 1)] #times of stim onset when choice was L and stim was R (contrast>0), so incorrect
    eventStim_stL100chR = timesStimOn_orig[np.logical_and(MoveandL100stim, choice_orig == -1)]
    eventStim_stL100chL = timesStimOn_orig[np.logical_and(MoveandL100stim, choice_orig == 1)] 
    
    #event == 'Feedback'
    timesFeedback_orig = one.load_dataset(eid, '_ibl_trials.feedback_times.npy')
    FeedbackType_orig = one.load_dataset(eid, '_ibl_trials.feedbackType.npy')
    FeedbackType = FeedbackType_orig[~np.isnan(timesStimOn_orig)] #Or should it be: FeedbackType_orig[~np.isnan(timesFeedback_orig)]? Sometimes a mismatch occurs
    #TO DO: Need to understand the discripency between event_CorrR and the following:
    #event_stRchR = times1stMove_orig[np.logical_and(MoveandRstim, choice_orig == -1)] #times of movement when choice was R and stim was R (contrast>0)
    #(i.e., mistmatch between shape(event_stLchL), e.g. =149, and shape(event_CorrL), e.g. =150.
    #How can the fdback be 1 if the mouse didn't move?; depending on result, may need to change the calculations in brainbox -> trials.py)
    event_CorrR = timesStimOn[np.logical_and(FeedbackType==1, contrast_R>0)] #Any non-zero R contrast that's correct
    event_CorrL = timesStimOn[np.logical_and(FeedbackType==1, contrast_L>0)] 
    event_IncorrR = timesStimOn[np.logical_and(FeedbackType==-1, contrast_R>0)] #Any non-zero R contrast that's correct
    event_IncorrL = timesStimOn[np.logical_and(FeedbackType==-1, contrast_L>0)] 

    timesFeedback = timesFeedback_orig[~np.isnan(timesFeedback_orig)]
    FeedbackType2 = FeedbackType_orig[~np.isnan(timesFeedback_orig)]
    event_FdbckCorr = timesFeedback[FeedbackType2 == 1] #Times when reward was given
    event_FdbckIncorr = timesFeedback[FeedbackType2 == -1] #Times when noise burst was given
    
    
    event_times = [event_times_right, event_times_left, event_times_right100, event_times_left100, 
                   event_times_0, event_times_Rchoice, event_times_Lchoice,
                   event_CorrR, event_CorrL, event_IncorrR, event_IncorrL,
                   eventMove_stR100chR, eventMove_stR100chL, eventMove_stL100chR, eventMove_stL100chL,
                   eventStim_stR100chR, eventStim_stR100chL, eventStim_stL100chR, eventStim_stL100chL,
                   event_FdbckCorr, event_FdbckCorr]
    event_Titles = ['event_times_right', 'event_times_left', 'event_times_right100', 'event_times_left100', 
                   'event_times_0', 'event_times_Rchoice', 'event_times_Lchoice',
                   'event_CorrR', 'event_CorrL', 'event_IncorrR', 'event_IncorrL',
                   'eventMove_stR100chR', 'eventMove_stR100chL', 'eventMove_stL100chR', 'eventMove_stL100chL',
                   'eventStim_stR100chR', 'eventStim_stR100chL', 'eventStim_stL100chR', 'eventStim_stL100chL',
                   'event_FdbckCorr', 'event_FdbckCorr']
    
    #TO DO: complete below:
    #if condition returns True, nothing happens; if condition returns False, AssertionError is raised:
    assert np.sum(np.isnan(event_times_left)) == 0
    assert np.sum(np.isnan(event_times_right)) == 0
    assert np.sum(np.isnan(event_times_0)) == 0
    assert np.sum(np.isnan(event_times_left100)) == 0
    assert np.sum(np.isnan(event_times_right100)) == 0
    assert np.sum(np.isnan(event_times_Rchoice)) == 0
    assert np.sum(np.isnan(event_times_Rchoice)) == 0
    assert np.sum(np.isnan(event_times_Lchoice)) == 0
    assert np.sum(np.isnan(event_CorrR)) == 0
    assert np.sum(np.isnan(event_CorrL)) == 0
    assert np.sum(np.isnan(event_IncorrR)) == 0
    assert np.sum(np.isnan(event_IncorrL)) == 0
    assert np.sum(np.isnan(base_line_times)) == 0
    
    
    activities = []
    BinnedSpikes = []
    #count = 0
    for etime in event_times:
        if len(etime)>0: # etime may be empty because some events don't occur in any trials        
            a, b = calculate_peths(spikes['times'], spikes['clusters'], np.array(clusterIDs),
                                   etime, pre_time=pre_time, post_time=post_time, smoothing=0, bin_size=binSzPeri)
        else:
            #a=Bunch({'means': np.array([]), 'stds': np.array([]), 'tscale': np.array([]), 'cscale': np.array([])})
            #a=Bunch({'means': np.array([nan]), 'stds': np.array([nan]), 'tscale': np.array([nan]), 'cscale': np.array([nan])})
            a=Bunch({'means': np.zeros([len(clusterIDs)]), 'stds': np.zeros([len(clusterIDs)]), 'tscale': np.zeros([len(clusterIDs)]), 'cscale': np.zeros([len(clusterIDs)])})
            b=np.empty([1, len(clusterIDs), int((post_time - (-pre_time))/binSzPeri)])
            b[:] = np.NaN
        
        activities.append(a)
        BinnedSpikes.append(b)
        #count += 1        
    
    for k, cluster in enumerate(clusterIDs):
        fr.append(spikes['times'][spikes['clusters'] == cluster].shape[0]/ spikes['times'][-1])
        Ntrials.append(len(event_times_left)+len(event_times_right) + len(event_times_0)) #Total # of behav. trials per eID (shows as per cluster)
        
        #This part calculates the f.r. in bins of 50 ms, then calculates the variance and Fano Factor:
        SpikesOfCluster = spikes['times'][spikes['clusters'] == cluster]
        time_bins = arange(0, spikes['times'][-1], 0.05)
        SpkIncrements50ms, _ = histogram(SpikesOfCluster, time_bins) #vector containing the # of spikes/counts in each 50 ms increment.

        FF50ms = SpkIncrements50ms.var()/SpkIncrements50ms.mean()
        FFofFR.append(FF50ms)        
           
        LabNumArray.append(LabNum)
        SpikeSortMethod.append(SpikeSorter)

        xvals.append(clusters['x'][clusters['metrics']['cluster_id'] == cluster])
        yvals.append(clusters['y'][clusters['metrics']['cluster_id'] == cluster])
        zvals.append(clusters['z'][clusters['metrics']['cluster_id'] == cluster])
        
        # #Now find the FR and Fano Factor around the event only, for each cluster:
        TimeVect0 = activities[0]['tscale']
        TimeVect.append(TimeVect0) #If the pre and post times for diff events are diff, will need to save the time vectors for all events later

        #Save all FR and FF over time for all events:
        FRoverT = [activities[x].means[activities[x].cscale == cluster] for x in range(0,len(event_times))]
        BinnedSpOfClust = [BinnedSpikes[x][:, activities[x].cscale == cluster, :] for x in range(0,len(event_times))]
        #Shape0 = [BinnedSpOfClust[x].shape[0] for x in range (0,len(event_times))] #need this for cases where there are no etime events in any trials
        for idx in range(0,len(event_times)):
            if size(BinnedSpOfClust[idx], 1) == 0: #there were no etime events 
                BinnedSpOfClust[idx] = np.empty([1, 1, int((post_time - (-pre_time))/binSzPeri)])
                BinnedSpOfClust[idx][:] = np.NaN
                FRoverT[idx] = np.empty([1, int((post_time - (-pre_time))/binSzPeri)])
                FRoverT[idx][:] = np.NaN
        #[i for i,x in enumerate(Shape0) if x==1]  # see here: https://stackoverflow.com/questions/9542738/python-find-in-list 
        #BinnedSpOfClust[Shape0==1]
        CountPerBin = [BinnedSpOfClust[x].reshape(BinnedSpOfClust[x].shape[0], BinnedSpOfClust[x].shape[2]) for x in range(0,len(event_times))]
        FFoverT = [np.nanvar(CountPerBin[x], axis=0)/np.nanmean(CountPerBin[x], axis=0) for x in range(0,len(event_times))]
        
        #Now the means of FR and FF, pre and post event, over all events:
        #FR_PostEvent = [float(np.nanmean(FRoverT[x][0,TimeVect0>0])) for x in range(0,len(event_times))]
        FR_PreEvent = [float(np.nanmean(FRoverT[x][0,TimeVect0<0])) for x in range(0,len(event_times))]
        FF_PreEvent = [float(np.nanmean(FFoverT[x][TimeVect0<0])) for x in range(0,len(event_times))]
        #FF_PostEvent = [float(np.nanmean(FFoverT[x][TimeVect0>0])) for x in range(0,len(event_times))]
        FR_PostEvent = [float(np.nanmean(FRoverT[x][0, np.logical_and(TimeVect0>0, TimeVect0<CapPostTime)])) for x in range(0,len(event_times))]
        FF_PostEvent = [float(np.nanmean(FFoverT[x][np.logical_and(TimeVect0>0, TimeVect0<CapPostTime)])) for x in range(0,len(event_times))]
       

        FRoverT_AllEv.append(FRoverT)   # for all events
        FFoverT_AllEv.append(FFoverT)  
        FRpre_AllEv.append(FR_PreEvent) 
        FRpost_AllEv.append(FR_PostEvent) 
        FFpre_AllEv.append(FF_PreEvent) 
        FFpost_AllEv.append(FF_PostEvent) 
        event_TitlesAll.append(event_Titles)
        
        
    #Save all the data:
    columns1=['eID', 'probeNum',  'dateInfo', 'Subject', 'LabID', 'clusterID',
              'AvgFR', 'AvgFFofFR', 'amps', 'peak_trough', 'Xloc', 'Yloc', 'Zloc',
              'RepeatedSite', 'SpikeSortMethod']
    columnsPeri=['eID', 'probeNum', 'dateInfo', 'Subject', 'LabID','clusterID', 
                 'Ntrial', 'FR_PreEvent', 'FR_PostEvent', 'FF_PreEvent', 'FF_PostEvent',
                 'TimeVect', 'FRoverT', 'FFoverT', 'event_Titles',
                 'RepeatedSite', 'SpikeSortMethod']
    if len(clusterIDs)>1:
        data1 = np.array([np.repeat(eid, len(clusterIDs)), np.repeat(probe, len(clusterIDs)), 
                          np.repeat(DateInfo, len(clusterIDs)), np.repeat(subj, len(clusterIDs)),
                          LabNumArray, np.array(clusterIDs), np.array(fr), np.array(FFofFR),
                          amps1[clusterIDs], ptt1[clusterIDs], np.squeeze(xvals), np.squeeze(yvals), np.squeeze(zvals),
                          np.repeat(RS_YorN, len(clusterIDs)), SpikeSortMethod])
        data1 = np.transpose(data1); #data1.reshape(len(clusterIDs),len(columns))
        
        #data frame for peri-event calculations:
        dataPeri = np.array([np.repeat(eid, len(clusterIDs)), np.repeat(probe, len(clusterIDs)), 
                             np.repeat(DateInfo, len(clusterIDs)), np.repeat(subj, len(clusterIDs)),
                             LabNumArray, np.array(clusterIDs), np.array(Ntrials),
                             FRpre_AllEv, FRpost_AllEv, FFpre_AllEv, FFpost_AllEv,
                             TimeVect, FRoverT_AllEv, FFoverT_AllEv, event_TitlesAll,
                             np.repeat(RS_YorN, len(clusterIDs)), SpikeSortMethod]) #np.array(FRoverT_ev0).reshape(76,44) for 1st eid
        dataPeri = np.transpose(dataPeri)
    elif len(clusterIDs)==1:
        data1 = np.array([eid, probe, DateInfo, subj, np.squeeze(LabNumArray), np.squeeze(np.array(clusterIDs)), np.squeeze(np.array(fr)), np.squeeze(np.array(FFofFR)),
                          np.squeeze(amps1[clusterIDs]), np.squeeze(ptt1[clusterIDs]), np.squeeze(xvals), np.squeeze(yvals), np.squeeze(zvals),
                          RS_YorN, np.squeeze(SpikeSortMethod)])
        data1 = data1.reshape(len(clusterIDs), len(columns1))

        #data frame for peri-event calculations:
        dataPeri = np.array([eid, probe, DateInfo, subj, np.squeeze(LabNumArray), np.squeeze(np.array(clusterIDs)), np.squeeze(np.array(Ntrials)),
                             np.squeeze(np.array(FRpre_AllEv)), np.squeeze(np.array(FRpost_AllEv)), np.squeeze(np.array(FFpre_AllEv)), np.squeeze(np.array(FFpost_AllEv)),
                             np.array(TimeVect), np.array(FRoverT_AllEv), np.array(FFoverT_AllEv),event_TitlesAll,
                             RS_YorN, np.squeeze(SpikeSortMethod)])
        dataPeri = dataPeri.reshape(len(clusterIDs), len(columnsPeri))

    #Data frame for session-averaged calculations:
    df = pd.DataFrame(data1, columns=columns1,
                      index=np.arange(len(ClusterFeatures), len(ClusterFeatures)+len(clusterIDs))) #index=np.arange(0,len(clusterIDs)))
    ClusterFeatures = ClusterFeatures.append(df) 
    
    #Data frame for perievent calculations:
    dfPeri = pd.DataFrame(dataPeri, columns=columnsPeri,
                      index=np.arange(len(PeriEventFRandFF), len(PeriEventFRandFF)+len(clusterIDs))) #index=np.arange(0,len(clusterIDs)))
    PeriEventFRandFF = PeriEventFRandFF.append(dfPeri)


# %% Find the center of mass of target brain regions and substract from xyz

#Find the planned repeated site trajectories
rep_traj = one.alyx.rest('trajectories', 'list', provenance='Planned', x=-2243, y=-2000, theta=15)
rep_traj = rep_traj[0]

# Create an insertion object from the trajectory
rep_ins = atlas.Insertion.from_dict(rep_traj, brain_atlas=ba)
# Find the xyz coords of tip of the insertion and the point where the insertion enters the brain
xyz = np.c_[rep_ins.tip, rep_ins.entry].T
# Find the brain regions along the insertion
brain_regions, _ = histology.get_brain_regions(xyz, brain_atlas=ba)

# Find the centre of mass of region
#Note: Brain regions are combined here, e.g., PPC
idx_reg = np.where(combine_regions(brain_regions.acronym) == regions)[0]
xyz_reg = brain_regions.xyz[idx_reg] #The coordinates we were planning to hit within the given brain region
centre_of_mass_reg = np.mean(xyz_reg, 0)

DeltaX = (np.array(ClusterFeatures['Xloc'], dtype = np.float64) - centre_of_mass_reg[0])*1e6
DeltaY = (np.array(ClusterFeatures['Yloc'], dtype = np.float64) - centre_of_mass_reg[1])*1e6
DeltaZ = (np.array(ClusterFeatures['Zloc'], dtype = np.float64) - centre_of_mass_reg[2])*1e6


# %% Save data and make plots

## If needed, use codes below to save DataFrame and save as .mat file for Matlab
if SaveFigs==1:
    # If needed, use codes below to save DataFrame and save as .mat file for Matlab
    ClusterFeatures.to_pickle(FIG_PATH+'/'+regions+'/'+regions +"_ClusterFeatures" + SaveStr + ".pkl")
    PeriEventFRandFF.to_pickle(FIG_PATH+'/'+regions+'/'+regions +"_PeriEventFRandFF" + SaveStr + ".pkl")
    # output = pd.read_pickle("ClusterFeatures_DF.pkl")
    # print(output)
    sio.savemat(FIG_PATH+'/'+regions+'/'+regions +'_ClusterFeatMat' + SaveStr + ".mat", {name: col.values for name, col in ClusterFeatures.items()})
    sio.savemat(FIG_PATH+'/'+regions+'/'+regions +'_PeriEventFRandFF' + SaveStr + ".mat", {name: col.values for name, col in PeriEventFRandFF.items()})

regionTitle = regions
cm = plt.get_cmap("turbo")  #viridis, vlag, Accent, cool, turbo

fig = plt.figure()
fig.suptitle(regions + ': LabID')
ax = fig.add_subplot(111, projection='3d') 
p = ax.scatter(DeltaX, DeltaY, DeltaZ, c=np.array(ClusterFeatures['LabID'], dtype = np.float64),cmap=cm, depthshade=False, s=4)
ax.set_xlabel('dX') #delta x (distance from center of mass of brain region)
ax.set_ylabel('dY')
ax.set_zlabel('dZ')  
cbar = plt.colorbar(p)
if SaveFigs==1:
    plt.savefig(join(FIG_PATH, regions, 'cluster 3D plots Lab ID' + SaveStr))

# #plot center of mass:
# p = ax.scatter([0],[0],[0], c='r', cmap=cm, depthshade=False, s=10)
# ax.plot([0], [0], [0], markerfacecolor='k', markeredgecolor='k', marker='o', markersize=5, alpha=0.6)
plt.show()

fig = plt.figure()
fig.suptitle(regions + ': Perievent FR')
ax = fig.add_subplot(111, projection='3d') 
FR_PostEvent0 = [PeriEventFRandFF['FR_PostEvent'][x][0] for x in range(0,len(PeriEventFRandFF))] #Get the FR of the 0th event for each cluster
p = ax.scatter(DeltaX, DeltaY, DeltaZ, c=np.array(FR_PostEvent0, dtype = np.float64),cmap=cm, depthshade=False, s=4)
ax.set_xlabel('dX') #delta x (distance from center of mass of brain region)
ax.set_ylabel('dY')
ax.set_zlabel('dZ')  
cbar = plt.colorbar(p)
if SaveFigs==1:
    plt.savefig(join(FIG_PATH, regions, 'cluster 3D plots post-event0 FR' + SaveStr))

fig = plt.figure()
fig.suptitle(regions + ': Perievent FF')
ax = fig.add_subplot(111, projection='3d') 
FF_PostEvent0 = [PeriEventFRandFF['FF_PostEvent'][x][0] for x in range(0,len(PeriEventFRandFF))] #Get the FR of the 0th event for each cluster
p = ax.scatter(DeltaX, DeltaY, DeltaZ, c=np.array(FF_PostEvent0, dtype = np.float64),cmap=cm, depthshade=False, s=4)
ax.set_xlabel('dX') #delta x (distance from center of mass of brain region)
ax.set_ylabel('dY')
ax.set_zlabel('dZ')  
cbar = plt.colorbar(p)
if SaveFigs==1:
    plt.savefig(join(FIG_PATH, regions, 'cluster 3D plots peri-event0 FF' + SaveStr))
    

fig = plt.figure()
fig.suptitle(regions + ': Session-Avg FR')
ax = fig.add_subplot(111, projection='3d') 
p = ax.scatter(DeltaX, DeltaY, DeltaZ, c=np.log(np.array(ClusterFeatures['AvgFR'], dtype = np.float64)),cmap=cm, depthshade=False, s=4)
ax.set_xlabel('dX') 
ax.set_ylabel('dY')
ax.set_zlabel('dZ')  
cbar = plt.colorbar(p)
if SaveFigs==1:
    plt.savefig(join(FIG_PATH, regions, 'cluster 3D plots log avg FR' + SaveStr))


fig = plt.figure()
fig.suptitle(regions + ': WF amp')
ax = fig.add_subplot(111, projection='3d') 
p = ax.scatter(DeltaX, DeltaY, DeltaZ, c=np.array(ClusterFeatures['amps'], dtype = np.float64), cmap=cm, depthshade=False, s=4)
ax.set_xlabel('dX') 
ax.set_ylabel('dY')
ax.set_zlabel('dZ')  
cbar = plt.colorbar(p)
if SaveFigs==1:
    plt.savefig(join(FIG_PATH, regions, 'cluster 3D plots WF amp' + SaveStr))


fig = plt.figure()
fig.suptitle(regions + ': WF peak_trough')
ax = fig.add_subplot(111, projection='3d') 
p = ax.scatter(DeltaX, DeltaY, DeltaZ, c=np.array(ClusterFeatures['peak_trough'], dtype = np.float64),cmap=cm, depthshade=False, s=4)
ax.set_xlabel('dX') 
ax.set_ylabel('dY')
ax.set_zlabel('dZ')  
cbar = plt.colorbar(p)
if SaveFigs==1:
    plt.savefig(join(FIG_PATH, regions, 'cluster 3D plots WF peak-trough' + SaveStr))


#plot histogram of the probes x,y, and z distances from the target
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
fig.suptitle(regions + ': Histogram of cluster locations')
sns.histplot(data=np.array(DeltaX), binwidth=10, ax=axes[0], legend= False)
axes[0].set_title('dX (um)')
sns.histplot(data=np.array(DeltaY), binwidth=10, ax=axes[1], legend= False)
axes[1].set_title('dY (um)')
sns.histplot(data=np.array(DeltaZ), binwidth=10, ax=axes[2], legend= False)
axes[2].set_title('dZ (um)')
sns.despine(trim=True)
plt.tight_layout()
if SaveFigs==1:
    plt.savefig(join(FIG_PATH, regions, 'Histogram of cluster positions_' + SaveStr))
    print('Figures saved')
    
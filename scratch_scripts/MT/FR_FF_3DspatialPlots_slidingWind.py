
# -*- coding: utf-8 -*-
"""
@author: Marsa Taheri

Note:
In this version of the code, we have the option of choosing a sliding window, causal or not. 
Also, the time resolution for FF and FR over time can be different.

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

from iblatlas import atlas
from iblatlas.atlas import AllenAtlas
from ibllib.pipes import histology
from iblutil.util import Bunch

from scipy.stats import wilcoxon#, ranksums, ttest_ind, ttest_rel
from brainbox.task._statsmodels import multipletests


one = ONE()
ba = AllenAtlas()

# Query repeated site trajectories
#traj = query()
#traj = query(behavior=True)
ListOfEIDs = eid_list()

# Initialize dataframe
ClusterFeatures = pd.DataFrame() #['eID', 'Lab ID','clusterID', 'AvgFR', 'AvgFF', 'amps', 'peak-trough', 'Xloc', 'Yloc', 'Zloc']
PeriEventFRandFF = pd.DataFrame() #['FR stimOn', 'FR '] FR and FF averaged over entire session

regions = 'LP' #[PPC', 'CA1', 'DG', 'LP', 'PO']
#binSzFRPeri_noSlide = 0.02 #bin size for calculating perievent FR (w/no sliding window)
#binSzPeri = binSzFRPeri_noSlide
binSzFRPeri = 0.04 #bin size for calculating perievent FR when using sliding window
n_slideFR = 2 #4 # Number of slides per bin for FR calculation when using sliding window
binSzFFPeri = 0.1 #bin size for calculating perievent FF
n_slideFF = 5 # Number of slides per bin for FF calculation when using sliding window
pre_time, post_time = 0.4, 0.8
CapPostTime = 0.4 #Time point at which to cap the avg FR/FF post-event; even though over time we have 0.8 s post-event, we can cap the analysis of avg post-event to 0.4 s 
Caus = 1; #should the FR and FF sliding window be calculated in a causal way (1) or not (0), i.e., each time point is the center of bin

SaveFigs=1 # To choose whether to save the figures & .pkl & .mat files at the end; to save, set equal to 1
SaveStr = '_SlidingWind_Pyk_restrTrials_010622' #string to have at the end of the figure/file names to be saved, e.g., the date

include_NonRS = 0 #Whether or not to include non-RS that pass the area; 0 for No, 1 for Yes.


# %% Functions (modified from others' codes)

# Retreives the trajectory for a list of eids:
def get_traj(ListOfEIDs):
    traj = query()
    tmp = []
    for t in traj:
        if t['session']['id'] in ListOfEIDs:
            tmp.append(t)
    traj = tmp
    
    return traj


# Calculates PETHS with a sliding window:
def cluster_peths_FR_FF_sliding(ts, align_times, pre_time=0.2, post_time=0.5, 
                                hist_win=0.1, N_SlidesPerWind = 5, causal=0):
    """
    Calcluate peri-event time histograms of one unit/cluster at a time; returns
    means and standard deviations of FR and FF over time for each time point 
    using a sliding window.

    :param ts: spike times of cluster (in seconds)
    :type ts: array-like
    :param align_times: times (in seconds) to align peths to
    :type align_times: array-like
    :param pre_time: time (in seconds) to precede align times in peth
    :type pre_time: float
    :param post_time: time (in seconds) to follow align times in peth
    :type post_time: float
    :param hist_win: width of time windows (in seconds) to bin spikes for each sliding window
    :type hist_win: float
    :param N_SlidesPerWind: The # of slides to do within each histogram window, i.e. increase in time resolution
    :type N_SlidesPerWind: float
    :param causal: whether or not to place time points at the end of each hist_win (1) or in the center (0)
    :type causal: float
    :return: FR_sorted, FR_STD_sorted, FF_sorted, TimeVect_sorted
    :rtype: np.array
    """
    #slidingStep = bin_size/n_slides
    epoch = [-pre_time, post_time]
    tshift = hist_win/N_SlidesPerWind 
    if causal==1: # Place time points at the end of each hist_win, i.e., only past events are taken into account.
        epoch[0] = epoch[0] - hist_win/2 #to start earlier since we're shifting the time later
        #epoch[1] = epoch[1] - hist_win/2
    
    
    FR_unsort, FRstd_unsort, FF_unsort, TimeVect = [], [], [], []
    for s in range(N_SlidesPerWind):
        ts_shift = ts[ts>(s*tshift)]
        PethsPerShift, BinnedSpPerShift = calculate_peths(ts_shift, np.ones_like(ts_shift), np.array([1]),
                                                          (align_times + s*tshift), pre_time=abs(epoch[0]), 
                                                          post_time=(epoch[1]- s*tshift), bin_size=hist_win,
                                                          smoothing=0, return_fr=False) 
        # To Do: The post_time=(epoch[1]- s*tshift) might need to become post_time=(epoch[1]- s*hist_win) or similar.
        
        CountPerBinPerShift = BinnedSpPerShift.reshape(BinnedSpPerShift.shape[0], BinnedSpPerShift.shape[2])
        #FR_PerTrialPerShift = CountPerBinPerShift/hist_win
        FR_TrialAvgPerShift = np.nanmean(CountPerBinPerShift, axis=0)/hist_win
        FR_TrialSTDPerShift = np.nanstd(CountPerBinPerShift, axis=0)/hist_win #stdev of firing rate; same as np.std(CountPerBinPerShift/hist_win, axis=0)
        FF_PerShift = np.nanvar(CountPerBinPerShift, axis=0)/np.nanmean(CountPerBinPerShift, axis=0)
        TimeVect_PerShift = PethsPerShift['tscale'] + s*tshift #np.arange(FR_PerShift.size) * hist_win + tshift*s #per slide

        # Place time points at the end of each hist_win (causal = 1),i.e., only past events are taken into account.
        # Otherwise, time points are at the center of the time bins (using 'calculate_peths')
        if causal==1: 
            TimeVect_PerShift = TimeVect_PerShift + hist_win/2
            
        # Append this shifted result with previous ones:
        FR_unsort = np.hstack((FR_unsort, FR_TrialAvgPerShift))
        FRstd_unsort = np.hstack((FRstd_unsort, FR_TrialSTDPerShift))
        FF_unsort = np.hstack((FF_unsort, FF_PerShift))
        TimeVect = np.hstack((TimeVect, TimeVect_PerShift)) #stacks the time vectors

    #Sort the time and FR vectors and convert lists to an np.array:
    TimeVect_sorted = np.sort(TimeVect)
    FR_sorted = np.array([x for _,x in sorted(zip(TimeVect, FR_unsort))])
    FR_STD_sorted = np.array([x for _,x in sorted(zip(TimeVect, FRstd_unsort))])
    FF_sorted = np.array([x for _,x in sorted(zip(TimeVect, FF_unsort))])
    
    return FR_sorted, FR_STD_sorted, FF_sorted, TimeVect_sorted


# # Used if firing rates are normalized:
# def normalise_act(activities, spikes, probe, clusterIDs, base_line_times):
#     activity_pre, _ = calculate_peths(spikes[probe]['times'], spikes[probe]['clusters'],
#                                       np.array(clusterIDs), base_line_times,
#                                       pre_time=0.4, post_time=-0.2, smoothing=0, bin_size=0.01) #-0.2 s means 200 ms BEFORE the stim.
#     baseline = np.mean(activity_pre.means, axis=1)

#     normed_activities = []
#     for a in activities:
#         normed_activities.append(((a.means.T - baseline) / (1 + baseline)).T)

#     return normed_activities


# %% Finds the trajectories to include in analysis (e.g., only RS ones, or also nearby probes):
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


# %% Loop through repeated site recordings and extract the data
PykiloYes, PykiloNo, NotWorking, NoGoodClustersInRegion = [],[],[],[]
NsignClust, NofAllClust = [], []
pVal_rTrial2AllClust=[]
for i in range(len(traj)):
    print('Processing repeated site recording %d of %d' % (i+1, len(traj)))
    fr, FFofFR, LabNumArray, SpikeSortMethod = [], [], [], []
    FR_during_RxnTime, FR_200msBeforeStim, RxnTimesPerTrial = [], [], []
    FR_RxnTime_restrTrials1, FR_RxnTime_restrTrials2, RxnTimes_rTrial1, RxnTimes_rTrial2 = [],[],[],[]
    FR_RxnTime_CorrTrials, FR_200mPreStim_CorrTrials = [], []
    FR_200msPre_rTrial1, FR_200msPre_rTrial2 = [], []
    FR_Trial400ms_rTrial1, FR_PostStim_rTrial1, FR_RpreMove_rTrial1, FR_LpreMove_rTrial1 = [], [], [], []
    pVal_TM, pVal_restrTrials2 = [],[] #signif_TM, pVal_TM, pVal_TM_corr = [],[],[] #for finding task-modulated clusters
    xvals, yvals, zvals =[], [], []
    TimeVect_FR, TimeVect_FF = [], []
    FRoverT_AllEv, FR_STD_overT_AllEv, FFoverT_AllEv = [], [], []
    FRpre_AllEv, FFpre_AllEv, FRpost_AllEv, FFpost_AllEv =[], [], [], []
    event_TitlesAll = []
    Ntrials=[]

    # Load in data
    eid = traj[i]['session']['id']
    probe = traj[i]['probe_name']
    DateInfo = traj[i]['session']['start_time'] #traj[i]['datetime']
    subj = traj[i]['session']['subject'] 

    #Is it a planned repeated site probe or not (e.g., nearby probe):
    if i<N_RSprobes:
        RS_YorN = 1
    else:
        RS_YorN = 0
        
    try:
        # #Try ks2 instead of Pykilosort:
        # spikes = one.load_object(eid, 'spikes', collection='alf/{}'.format(probe), revision='') #, revision='' needed or not?? otherwise uses latest revision
        # clusters = one.load_object(eid, 'clusters', collection='alf/{}'.format(probe), revision='')
        # channels = bbone.load_channel_locations(eid, one=one, probe=probe, aligned=True)
        # clusters = bbone.merge_clusters_channels(dic_clus={probe: clusters}, channels=channels)[probe]
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

        # #? clusterIDs  = clusters['metrics']['cluster_id'][BrainRegionsInProbe == regions][clusters['metrics']['label'] == 1]
        PykiloYes.append(traj[i]['session']['subject'])
        SpikeSorter = 'Pykilosort'
        # PykiloNo.append(traj[i]['session']['subject'])
        # SpikeSorter = 'KS2'
        
    except:
        try:
            #Try ks2 instead of Pykilosort:
            spikes = one.load_object(eid, 'spikes', collection='alf/{}'.format(probe), revision='') #, revision='' needed or not?? otherwise uses latest revision
            clusters = one.load_object(eid, 'clusters', collection='alf/{}'.format(probe), revision='')
            channels = bbone.load_channel_locations(eid, one=one, probe=probe, aligned=True)
            clusters = bbone.merge_clusters_channels(dic_clus={probe: clusters}, channels=channels)[probe]
            # #Try Pykilosort: 
            # spikes = one.load_object(eid, 'spikes', collection='alf/{}/pykilosort'.format(probe))
            # clusters = one.load_object(eid, 'clusters', collection='alf/{}/pykilosort'.format(probe))
            # channels = bbone.load_channel_locations(eid, one=one, probe=probe, aligned=True)
            # clusters = bbone.merge_clusters_channels(dic_clus={probe: clusters}, channels=channels)[probe]
            
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
            # PykiloYes.append(traj[i]['session']['subject'])
            # SpikeSorter = 'Pykilosort'
        
        except:
            NotWorking.append(traj[i]['session']['subject'])
            print(traj[i]['session']['subject'], ' !!No Clusters!!')
            continue

    # #May not need this part below (but was it useful, and why included in others' codes?):
    # # Get coordinates of micro-manipulator and histology
    # hist = one.alyx.rest('trajectories', 'list', provenance='Histology track',
    #                  probe_insertion=traj[i]['probe_insertion'])
    # manipulator = one.alyx.rest('trajectories', 'list', provenance='Micro-manipulator',
    #                             probe_insertion=traj[i]['probe_insertion'])
    # if (len(hist) == 0) or (len(manipulator) == 0):
    #     continue


    # Find lab name, and associate the lab name with a number:
    LabName = traj[i]['session']['lab']
    lab_number_map, _, _ = labs() #lab_number_map, institution_map, lab_colors = labs()
    LabNum = int(lab_number_map.get(LabName)[-1]) #note: Lab 0 is really Lab 10, which is UCLA
    

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
    #(Could potentially replace the following with the find_trial_IDs function)
    #event == 'Stim' 
    timesStimOn_orig = one.load_dataset(eid, '_ibl_trials.stimOn_times.npy') #one.load(eid, dataset_types=['trials.stimOn_times'])[0]
    #base_line_times = one.load_dataset(eid, '_ibl_trials.stimOn_times.npy')[0] #one.load(eid, dataset_types=['trials.stimOn_times'])[0]
    #base_line_times = base_line_times[~np.isnan(base_line_times)]
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
    #Aligned to movement, but for any nonzero contrast:
    MoveandRstim = np.logical_and(contrast_R_orig > 0, StimWithChoice) #events when stim was R (contrast>0) and there was a choice ultimately
    MoveandLstim = np.logical_and(contrast_L_orig > 0, StimWithChoice) 
    eventMove_stRchR = times1stMove_orig[np.logical_and(MoveandRstim, choice_orig == -1)] #times of movement when choice was R and stim was R (contrast>0)
    eventMove_stRchL = times1stMove_orig[np.logical_and(MoveandRstim, choice_orig == 1)] #times of movement when choice was L and stim was R (contrast>0), so incorrect
    eventMove_stLchR = times1stMove_orig[np.logical_and(MoveandLstim, choice_orig == -1)]
    eventMove_stLchL = times1stMove_orig[np.logical_and(MoveandLstim, choice_orig == 1)] 
    

    
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

    
    #Can also extract these from the one.load_dataset(): _ibl_trials.probabilityLeft.npy; _ibl_trials.response_times.npy
    #!!! Can later compare prob left vs. right pre-movement!
    

    ### TASK-MODULATION TEST 1 ###
    # The following section is for time-warping pre-movement vs. pre-stim analyses:
    #(Note: timesStimOn_orig changed to timesStimOn due to last NaN stim sometimes; hence, defined times1stMove_orig2)

    #Reaction time (i.e., time from stim On until 1st movement):
    times1stMove_orig2 = times1stMove_orig[~np.isnan(timesStimOn_orig)]
    RxnTimes = times1stMove_orig2-timesStimOn #include the _orig so that any NaNs stay nan
    #TrialsToExclude = np.where(np.isnan(RxnTimes)) #either NaN movement or stim
    RxnTimes_fullTrial = RxnTimes[~np.isnan(RxnTimes)]
    times1stMove_fullTrial = times1stMove_orig2[~np.isnan(RxnTimes)]
    timesStimOn_fullTrial = timesStimOn[~np.isnan(RxnTimes)]

    #Same as above, but restrict to only correct trials & constrast>0:
    RxnTimes_CorrTrialWithNan = RxnTimes[np.logical_or(np.logical_and(FeedbackType==1, contrast_R>0),
                                                np.logical_and(FeedbackType==1, contrast_L>0))]
    RxnTimes_CorrTrial = RxnTimes_CorrTrialWithNan[~np.isnan(RxnTimes_CorrTrialWithNan)]
    times1stMove_CorrTrial = times1stMove_orig2[np.logical_or(np.logical_and(FeedbackType==1, contrast_R>0),
                                                             np.logical_and(FeedbackType==1, contrast_L>0))]  
    times1stMove_CorrTrial = times1stMove_CorrTrial[~np.isnan(RxnTimes_CorrTrialWithNan)]                
    timesStimOn_CorrTrial = timesStimOn[np.logical_or(np.logical_and(FeedbackType==1, contrast_R>0),
                                                           np.logical_and(FeedbackType==1, contrast_L>0))]
    timesStimOn_CorrTrial = timesStimOn_CorrTrial[~np.isnan(RxnTimes_CorrTrialWithNan)]
                                                    
    #In the case below, restrict trials to:
    #(1) Up to only 200 ms pre-movement (so not the long waiting periods):
    timesStart = timesStimOn
    timesStart[RxnTimes>0.2] = times1stMove_orig2[RxnTimes>0.2] - 0.2
    #(2) Trials with >50 ms between stim and movement:
    timesStart_restrTrials1 = timesStart[RxnTimes>0.05]
    times1stMove_restrTrials1 = times1stMove_orig2[RxnTimes>0.05]
    RxnTimes_restrTrials1 = times1stMove_restrTrials1 - timesStart_restrTrials1
    #(3) trials that were correct choices and constrast>0 (from (1), then apply (2) again):
    timesStart_restrTrialsTemp = timesStart[np.logical_or(np.logical_and(FeedbackType==1, contrast_R>0),
                                                          np.logical_and(FeedbackType==1, contrast_L>0))]
    times1stMove_restrTrialsTemp = times1stMove_orig2[np.logical_or(np.logical_and(FeedbackType==1, contrast_R>0),
                                                                   np.logical_and(FeedbackType==1, contrast_L>0))]
    RxnTimesTemp = times1stMove_restrTrialsTemp - timesStart_restrTrialsTemp
    timesStart_restrTrials2 = timesStart_restrTrialsTemp[RxnTimesTemp>0.05]
    times1stMove_restrTrials2 = times1stMove_restrTrialsTemp[RxnTimesTemp>0.05]
    RxnTimes_restrTrials2 = times1stMove_restrTrials1 - timesStart_restrTrials1
    #The above gives the 2 restriction options, one with correct trials and non-zero contrasts only:
    #timesStart_restrTrials1, times1stMove_restrTrials1, RxnTimes_restrTrials1
    #timesStart_restrTrials2, times1stMove_restrTrials2, RxnTimes_restrTrials2
    #Note: with the above, the np.isnan(RxnTimes) trials should be gone already since RxnTimes>0.05 was applied; double check.


    ### TASK-MODULATION TEST 2 ###
    # The following section is for Left vs. Right pre-movement:
    # Note: Restricted to only correct trials & constrast>0:
    RxnTimes_RightCorrWithNan = RxnTimes[np.logical_and(FeedbackType==1, contrast_R>0)]
    RxnTimes_RightCorr = RxnTimes_RightCorrWithNan[~np.isnan(RxnTimes_RightCorrWithNan)] #not used ATM
    RxnTimes_LeftCorrWithNan = RxnTimes[np.logical_and(FeedbackType==1, contrast_L>0)]
    RxnTimes_LeftCorr = RxnTimes_LeftCorrWithNan[~np.isnan(RxnTimes_LeftCorrWithNan)] #not used ATM
    times1stMove_RightCorr = times1stMove_orig2[np.logical_and(FeedbackType==1, contrast_R>0)]
    times1stMove_RightCorr = times1stMove_RightCorr[~np.isnan(RxnTimes_RightCorrWithNan)]       
    times1stMove_LeftCorr = times1stMove_orig2[np.logical_and(FeedbackType==1, contrast_L>0)]
    times1stMove_LeftCorr = times1stMove_LeftCorr[~np.isnan(RxnTimes_LeftCorrWithNan)]               
    #later make more precise by considering the time between L and R
    
    
    

# idxs = np.bitwise_and(spike_times >= timesStimOn[t], spike_times <= times1stMove[t])
# i_spikes = spike_times[idxs]
# i_clusters = spike_clusters[idxs]
# # filter spikes outside of the loop
# idxs = np.bitwise_and(spike_times >= np.min(align_times) - (n_bins_pre + 1) * bin_size,
#                       spike_times <= np.max(align_times) + (n_bins_post + 1) * bin_size)
# idxs = np.bitwise_and(idxs, np.isin(spike_clusters, cluster_ids))
# spike_times = spike_times[idxs]
# spike_clusters = spike_clusters[idxs]

    
    event_times = [event_times_right, event_times_left, event_times_right100, event_times_left100, 
                   event_times_0, event_times_Rchoice, event_times_Lchoice,
                   event_CorrR, event_CorrL, event_IncorrR, event_IncorrL,
                   eventMove_stR100chR, eventMove_stR100chL, eventMove_stL100chR, eventMove_stL100chL,
                   eventStim_stR100chR, eventStim_stR100chL, eventStim_stL100chR, eventStim_stL100chL,
                   event_FdbckCorr, event_FdbckIncorr,
                   eventMove_stRchR, eventMove_stRchL, eventMove_stLchL, eventMove_stLchR] #!!! repeated!
    event_Titles = ['event_times_right', 'event_times_left', 'event_times_right100', 'event_times_left100', 
                   'event_times_0', 'event_times_Rchoice', 'event_times_Lchoice',
                   'event_CorrR', 'event_CorrL', 'event_IncorrR', 'event_IncorrL',
                   'eventMove_stR100chR', 'eventMove_stR100chL', 'eventMove_stL100chR', 'eventMove_stL100chL',
                   'eventStim_stR100chR', 'eventStim_stR100chL', 'eventStim_stL100chR', 'eventStim_stL100chL',
                   'event_FdbckCorr', 'event_FdbckIncorr',
                   'eventMove_stRchR', 'eventMove_stRchL', 'eventMove_stLchL', 'eventMove_stLchR']
    
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
    #assert np.sum(np.isnan(base_line_times)) == 0
    
    for k, cluster in enumerate(clusterIDs):
        fr.append(spikes['times'][spikes['clusters'] == cluster].shape[0]/ spikes['times'][-1])
        Ntrials.append(len(event_times_left)+len(event_times_right) + len(event_times_0)) #Total # of behav. trials per eID (shows as per cluster)
        
        #This part calculates the f.r. in bins of 50 ms, then calculates the variance and Fano Factor:
        SpikesOfCluster = spikes['times'][spikes['clusters'] == cluster]
        time_bins = arange(0, spikes['times'][-1], 0.05)
        SpkIncrements50ms, _ = histogram(SpikesOfCluster, time_bins) #vector containing the # of spikes/counts in each 50 ms increment.

        FF50ms = SpkIncrements50ms.var()/SpkIncrements50ms.mean()
        FFofFR.append(FF50ms) #This is similar to the coefficient of variation; different from the FF calculations below.
           
        LabNumArray.append(LabNum)
        SpikeSortMethod.append(SpikeSorter)
    
        xvals.append(clusters['x'][clusters['metrics']['cluster_id'] == cluster])
        yvals.append(clusters['y'][clusters['metrics']['cluster_id'] == cluster])
        zvals.append(clusters['z'][clusters['metrics']['cluster_id'] == cluster])
        

        #Q: Is it faster to make 1 for loop for all these statements?
        spike_times_per_cluster = spikes['times'][spikes['clusters'] == cluster]
        FR_200msBeforeStim_perClust = [sum(np.bitwise_and(spike_times_per_cluster >= timesStimOn_fullTrial[t]-0.2, spike_times_per_cluster < timesStimOn_fullTrial[t]))/0.2 
                             for t in range(0,len(timesStimOn_fullTrial))]
        FR_during_RxnTime_perClust = [sum(np.bitwise_and(spike_times_per_cluster >= timesStimOn_fullTrial[t], spike_times_per_cluster <= times1stMove_fullTrial[t]))/RxnTimes_fullTrial[t] 
                             for t in range(0,len(timesStimOn_fullTrial))]
        FR_200msPreStim_Corr = [sum(np.bitwise_and(spike_times_per_cluster >= timesStimOn_CorrTrial[t]-0.2, spike_times_per_cluster < timesStimOn_CorrTrial[t]))/0.2 
                             for t in range(0,len(timesStimOn_CorrTrial))]
        FR_during_RxnTime_Corr = [sum(np.bitwise_and(spike_times_per_cluster >= timesStimOn_CorrTrial[t], spike_times_per_cluster <= times1stMove_CorrTrial[t]))/RxnTimes_CorrTrial[t] 
                             for t in range(0,len(timesStimOn_CorrTrial))]
        FR_200msPreStim_restrTrials1 = [sum(np.bitwise_and(spike_times_per_cluster >= timesStart_restrTrials1[t]-0.2, spike_times_per_cluster < timesStart_restrTrials1[t]))/0.2 
                             for t in range(0,len(timesStart_restrTrials1))]
        FR_RxnTime_perClust_restrTrials1 = [sum(np.bitwise_and(spike_times_per_cluster >= timesStart_restrTrials1[t], spike_times_per_cluster <= times1stMove_restrTrials1[t]))/RxnTimes_restrTrials1[t] 
                             for t in range(0,len(timesStart_restrTrials1))]
        FR_200msPreStim_restrTrials2 = [sum(np.bitwise_and(spike_times_per_cluster >= timesStart_restrTrials2[t]-0.2, spike_times_per_cluster < timesStart_restrTrials2[t]))/0.2 
                             for t in range(0,len(timesStart_restrTrials2))]
        FR_RxnTime_perClust_restrTrials2 = [sum(np.bitwise_and(spike_times_per_cluster >= timesStart_restrTrials2[t], spike_times_per_cluster <= times1stMove_restrTrials2[t]))/RxnTimes_restrTrials2[t] 
                             for t in range(0,len(timesStart_restrTrials2))]        
        FR_400msPostStim_restrTrials1 = [sum(np.bitwise_and(spike_times_per_cluster >= timesStimOn_CorrTrial[t], spike_times_per_cluster < (timesStimOn_CorrTrial[t] + 0.4)))/0.4 
                             for t in range(0,len(timesStimOn_CorrTrial))]
        FR_100msPostStim_restrTrials1 = [sum(np.bitwise_and(spike_times_per_cluster >= timesStimOn_CorrTrial[t]+0.05, spike_times_per_cluster < (timesStimOn_CorrTrial[t] + 0.15)))/0.1 
                             for t in range(0,len(timesStimOn_CorrTrial))]
        FR_RightPreMove_restrTrials1 = [sum(np.bitwise_and(spike_times_per_cluster >= times1stMove_RightCorr[t]-0.1, spike_times_per_cluster < (times1stMove_RightCorr[t] + 0.05)))/0.15
                             for t in range(0,len(times1stMove_RightCorr))]
        FR_LeftPreMove_restrTrials1 = [sum(np.bitwise_and(spike_times_per_cluster >= times1stMove_LeftCorr[t]-0.1, spike_times_per_cluster < (times1stMove_LeftCorr[t] + 0.05)))/0.15
                             for t in range(0,len(times1stMove_LeftCorr))]        



        FR_200msBeforeStim.append(FR_200msBeforeStim_perClust)
        FR_during_RxnTime.append(FR_during_RxnTime_perClust)
        FR_200mPreStim_CorrTrials.append(FR_200msPreStim_Corr)
        FR_RxnTime_CorrTrials.append(FR_during_RxnTime_Corr)
        FR_200msPre_rTrial1.append(FR_200msPreStim_restrTrials1)
        FR_RxnTime_restrTrials1.append(FR_RxnTime_perClust_restrTrials1)
        FR_200msPre_rTrial2.append(FR_200msPreStim_restrTrials2)
        FR_RxnTime_restrTrials2.append(FR_RxnTime_perClust_restrTrials2)
        RxnTimesPerTrial.append(RxnTimes) #this is not only the full trials, i.e., nans may be included
        RxnTimes_rTrial1.append(RxnTimes_restrTrials1)
        RxnTimes_rTrial2.append(RxnTimes_restrTrials2)
        #[mean(FR_during_RxnTime_perClust), mean(FR_200msBeforeStim_perClust)]
        #[std(FR_during_RxnTime_perClust)/sqrt(Ntrials), std(FR_200msBeforeStim_perClust)/sqrt(Ntrials)]
        FR_Trial400ms_rTrial1.append(FR_400msPostStim_restrTrials1)
        FR_PostStim_rTrial1.append(FR_100msPostStim_restrTrials1)
        FR_RpreMove_rTrial1.append(FR_RightPreMove_restrTrials1)
        FR_LpreMove_rTrial1.append(FR_LeftPreMove_restrTrials1)
        
        

        #Test whether the unit is significantly modulated during the reaction time; based off of code in 
        #https://github.com/int-brain-lab/ibllib/blob/master/brainbox/task/closed_loop.py but FR rather than spike count
        if np.sum(np.array(FR_200msBeforeStim_perClust) - np.array(FR_during_RxnTime_perClust)) == 0:
            p_val_TaskMod = 1
            stats_TaskMod = 0
        else:
            stats_TaskMod, p_val_TaskMod = wilcoxon(FR_200msBeforeStim_perClust, FR_during_RxnTime_perClust)
            #stats_TaskMod, p_val_TaskMod = wilcoxon(np.array(FR_200msBeforeStim_perClust), np.array(FR_during_RxnTime_perClust))
            
        # # Perform FDR correction for multiple testing; this only makes sense with multiple clusters, so not needed here:
        # significance, p_val_TaskMod_corrected, _, _ = multipletests(p_val_TaskMod, alpha=0.05, method='fdr_bh')
        # if significance==True:
        # #     significant_units.append(cluster)
        #     signif_TM.append(1)
        # else:
        #     signif_TM.append(0)
 
        if np.sum(np.array(FR_200msPreStim_restrTrials2) - np.array(FR_RxnTime_perClust_restrTrials2)) == 0:
            p_val_restrTrials2 = 1
            stats_restrTrials2 = 0
        else:
            stats_restrTrials2, p_val_restrTrials2 = wilcoxon(FR_200msPreStim_restrTrials2, FR_RxnTime_perClust_restrTrials2)


        pVal_TM.append(p_val_TaskMod)
        pVal_restrTrials2.append(p_val_restrTrials2)
        #pVal_TM_corr.append(p_val_TaskMod_corrected)



        #Make calculations for FR and FF traces over time using a sliding window:
        #First, need to find cases for which no events occur, and modify such that calculations are possible:
        maskNoEvent = [1]*len(event_times) #[]
        for idx in range(0,len(event_times)):
            if size(event_times[idx]) == 0:
                maskNoEvent[idx]= event_times[idx]#0 #1 #.append(1)
                event_times[idx] = np.array([1], dtype=float64)
                
        #Calculate the FR using a smaller bin size of binSzFRPeri (~20 ms) and with sliding window for 1 cluster at a time, for all events:
        ActivitySmallBin = [cluster_peths_FR_FF_sliding(spikes['times'][spikes['clusters'] == cluster], 
                                                      event_times[x], pre_time=pre_time, post_time=post_time,
                                                      hist_win=binSzFRPeri, N_SlidesPerWind = n_slideFR, causal=Caus) for x in range(0,len(event_times))]
        FRoverT = [ActivitySmallBin[x][0] for x in range(0,len(event_times))]
        FR_STD_overT = [ActivitySmallBin[x][1] for x in range(0,len(event_times))]
        TimeVect_FR0 = ActivitySmallBin[0][3] #the 0th event, the 4th np.array (with idx=3) which is the time vector
        TimeVect_FR.append(TimeVect_FR0)

        #Calculate the FF using a larger bin size of binSzFFPeri (~100 ms) and with sliding window for 1 cluster at a time, for all events:
        ActivitySlideLargeBin = [cluster_peths_FR_FF_sliding(spikes['times'][spikes['clusters'] == cluster], 
                                                      event_times[x], pre_time=pre_time, post_time=post_time,
                                                      hist_win=binSzFFPeri, N_SlidesPerWind = n_slideFF, causal=Caus) for x in range(0,len(event_times))]
        FFoverT = [ActivitySlideLargeBin[x][2] for x in range(0,len(event_times))]
        TimeVect_FF0 = ActivitySlideLargeBin[0][3] #the 0th event, the 4th np.array (with idx=3) which is the time vector
        TimeVect_FF.append(TimeVect_FF0)
    
        # test=[d for d, s in zip(FRoverT, maskNoEvent) if s] #keep the FRoverT cases where maskNoEvent was true, i.e., 1 
        for idx in range(0, len(event_times)):
            if size(maskNoEvent[idx]) == 0: #cases where there were no events
                event_times[idx] = maskNoEvent[idx]      
                #FFoverT[idx] = np.empty([1, 1, int((post_time - (-pre_time))/binSzFFPeri)])
                FFoverT[idx][:] = np.NaN
                #FRoverT[idx] = np.empty([1, int((post_time - (-pre_time))/binSzFRPeri)])
                FRoverT[idx][:] = np.NaN
                FR_STD_overT[idx] = FRoverT[idx]
                
   
        #Now the means of FR and FF, pre and post event, over all events:
        FR_PreEvent = [float(np.nanmean(FRoverT[x][TimeVect_FR0<0])) for x in range(0,len(event_times))]
        FF_PreEvent = [float(np.nanmean(FFoverT[x][TimeVect_FF0<0])) for x in range(0,len(event_times))]
        FR_PostEvent = [float(np.nanmean(FRoverT[x][np.logical_and(TimeVect_FR0>0, TimeVect_FR0<CapPostTime)])) for x in range(0,len(event_times))]
        FF_PostEvent = [float(np.nanmean(FFoverT[x][np.logical_and(TimeVect_FF0>0, TimeVect_FF0<CapPostTime)])) for x in range(0,len(event_times))]
       

        FRoverT_AllEv.append(FRoverT)   # for all events
        FR_STD_overT_AllEv.append(FR_STD_overT)
        FFoverT_AllEv.append(FFoverT)  
        FRpre_AllEv.append(FR_PreEvent) 
        FRpost_AllEv.append(FR_PostEvent) 
        FFpre_AllEv.append(FF_PreEvent) 
        FFpost_AllEv.append(FF_PostEvent) 
        event_TitlesAll.append(event_Titles)
        

    
    #Correct the p-value to find which units are significantly modulated by the task (before the movement onset):
    significance, pVal_TM_corrected, _, _ = multipletests(pVal_TM, alpha=0.05, method='fdr_bh') #Note: correction applied to all clusters within each 1 recording
    NsignClust.append(sum(significance))
    NofAllClust.append(len(significance))
    significance_restrTrials2, pVal_restrTrials2_corr, _, _ = multipletests(pVal_restrTrials2, alpha=0.05, method='fdr_bh') #Note: correction applied to all clusters within each 1 recording
    print('%s: %d significant clusters out of %d' % (subj, sum(significance), len(significance)))
        
    #Save all the data:
    columns1=['eID', 'probeNum',  'dateInfo', 'Subject', 'LabID', 'clusterID',
              'AvgFR', 'AvgFFofFR', 'amps', 'peak_trough', 'Xloc', 'Yloc', 'Zloc',
              'RxnTimesPerTrial', 'RepeatedSite', 'SpikeSortMethod']
    columnsPeri=['eID', 'probeNum', 'dateInfo', 'Subject', 'LabID','clusterID', 
                 'Ntrial', 'FR_PreEvent', 'FR_PostEvent', 'FF_PreEvent', 'FF_PostEvent',
                 'TimeVect_FR', 'TimeVect_FF', 'FRoverT', 'FR_STD_overT', 'FFoverT', 
                 'event_Titles', 'FR_during_RxnTime','FR_200msBeforeStim',
                 'FR_RxnTime_CorrTrials', 'FR_200mPreStim_CorrTrials',
                 'FR_RxnTime_restrTrials1', 'FR_RxnTime_restrTrials2', 'FR_200msPre_rTrial1', 'FR_200msPre_rTrial2',
                 'RxnTimes_rTrial1', 'RxnTimes_rTrial2','FR_Trial400ms_rTrial1', 'FR_PostStim_rTrial1',
                 'FR_RpreMove_rTrial1', 'FR_LpreMove_rTrial1',
                 'pVal_TM', 'pVal_TM_corrected', 'pVal_restrTrials2', 'pVal_restrTrials2_corr',
                 'RepeatedSite', 'SpikeSortMethod']
    if len(clusterIDs)>1:
        data1 = np.array([np.repeat(eid, len(clusterIDs)), np.repeat(probe, len(clusterIDs)), 
                          np.repeat(DateInfo, len(clusterIDs)), np.repeat(subj, len(clusterIDs)),
                          LabNumArray, np.array(clusterIDs), np.array(fr), np.array(FFofFR),
                          amps1[clusterIDs], ptt1[clusterIDs], np.squeeze(xvals), np.squeeze(yvals), np.squeeze(zvals),
                          RxnTimesPerTrial, np.repeat(RS_YorN, len(clusterIDs)), SpikeSortMethod])
        data1 = np.transpose(data1); #data1.reshape(len(clusterIDs),len(columns))
        
        #data frame for peri-event calculations:
        dataPeri = np.array([np.repeat(eid, len(clusterIDs)), np.repeat(probe, len(clusterIDs)), 
                             np.repeat(DateInfo, len(clusterIDs)), np.repeat(subj, len(clusterIDs)),
                             LabNumArray, np.array(clusterIDs), np.array(Ntrials),
                             FRpre_AllEv, FRpost_AllEv, FFpre_AllEv, FFpost_AllEv,
                             TimeVect_FR, TimeVect_FF, FRoverT_AllEv, FR_STD_overT_AllEv, FFoverT_AllEv, 
                             event_TitlesAll, FR_during_RxnTime, FR_200msBeforeStim, 
                             FR_RxnTime_CorrTrials, FR_200mPreStim_CorrTrials,
                             FR_RxnTime_restrTrials1, FR_RxnTime_restrTrials2, FR_200msPre_rTrial1, FR_200msPre_rTrial2,
                             RxnTimes_rTrial1, RxnTimes_rTrial2, FR_Trial400ms_rTrial1, FR_PostStim_rTrial1,
                             FR_RpreMove_rTrial1, FR_LpreMove_rTrial1,
                             pVal_TM, pVal_TM_corrected, pVal_restrTrials2, pVal_restrTrials2_corr,
                             np.repeat(RS_YorN, len(clusterIDs)), SpikeSortMethod]) #np.array(FRoverT_ev0).reshape(76,44) for 1st eid
        dataPeri = np.transpose(dataPeri)
    elif len(clusterIDs)==1:
        data1 = np.array([eid, probe, DateInfo, subj, np.squeeze(LabNumArray), np.squeeze(np.array(clusterIDs)), np.squeeze(np.array(fr)), np.squeeze(np.array(FFofFR)),
                          np.squeeze(amps1[clusterIDs]), np.squeeze(ptt1[clusterIDs]), np.squeeze(xvals), np.squeeze(yvals), np.squeeze(zvals),
                          RxnTimesPerTrial, RS_YorN, np.squeeze(SpikeSortMethod)])
        data1 = data1.reshape(len(clusterIDs), len(columns1))

        #data frame for peri-event calculations:
        dataPeri = np.array([eid, probe, DateInfo, subj, np.squeeze(LabNumArray), np.squeeze(np.array(clusterIDs)), np.squeeze(np.array(Ntrials)),
                             np.squeeze(np.array(FRpre_AllEv)), np.squeeze(np.array(FRpost_AllEv)), np.squeeze(np.array(FFpre_AllEv)), np.squeeze(np.array(FFpost_AllEv)),
                             np.array(TimeVect_FR), np.array(TimeVect_FF), 
                             np.array(FRoverT_AllEv), np.array(FR_STD_overT_AllEv), np.array(FFoverT_AllEv),
                             event_TitlesAll, FR_during_RxnTime, FR_200msBeforeStim, 
                             FR_RxnTime_CorrTrials, FR_200mPreStim_CorrTrials,
                             FR_RxnTime_restrTrials1, FR_RxnTime_restrTrials2, FR_200msPre_rTrial1, FR_200msPre_rTrial2,
                             RxnTimes_rTrial1, RxnTimes_rTrial2, FR_Trial400ms_rTrial1, FR_PostStim_rTrial1,
                             FR_RpreMove_rTrial1, FR_LpreMove_rTrial1,
                             pVal_TM, pVal_TM_corrected, pVal_restrTrials2, pVal_restrTrials2_corr,
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

    #To do a multiple comparison across all clusters, save all p-values first:
    pVal_rTrial2AllClust.append(np.array(pVal_restrTrials2)) #each pVal_restrTrials2 is from 1 recordings, so append them all here

#Then, correct the p-value to find which units are significantly modulated by the task (before the movement onset):
#Convert the list of arrays (each array for 1 recordings) into 1 list or 1 array, where each element is the p-value of each cluster:
flat_list_ps = [item for subarray in pVal_rTrial2AllClust for item in subarray]  # which is equivalent to the following:
# flat_list_ps = []
# for subarray in pVal_rTrial2AllClust:
#     for item in subarray:
#         flat_list.append(item)
Pvals_array = np.array(flat_list_ps)
signifAll, pVal_TM_correctedAll, _, _ = multipletests(Pvals_array, alpha=0.05, method='fdr_bh') #Note: correction applied to ?
#signifAll, pVal_TM_correctedAll, _, _ = multipletests(pVal_rTrial2AllClust, alpha=0.05, method='fdr_bh') #Note: correction applied to ?
NsignClustAll = sum(signifAll)
NofAllClustAll = len(signifAll)
print('%s: %d significant clusters out of %d' % (subj, sum(signifAll), len(signifAll)))
 
    


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
    ClusterFeatures.to_pickle(FIG_PATH+'/'+regions+'/'+regions +"_ClusterFeatures_DF" + SaveStr + ".pkl")
    PeriEventFRandFF.to_pickle(FIG_PATH+'/'+regions+'/'+regions +"_PeriEventFRandFF_DF" + SaveStr + ".pkl")
    # output = pd.read_pickle("ClusterFeatures_DF.pkl")
    # print(output)
    sio.savemat(FIG_PATH+'/'+regions+'/'+regions +"_ClusterFeatMat" + SaveStr + ".mat", {name: col.values for name, col in ClusterFeatures.items()})
    sio.savemat(FIG_PATH+'/'+regions+'/'+regions +"_PeriEventFRandFF" + SaveStr + ".mat", {name: col.values for name, col in PeriEventFRandFF.items()})

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
ax.set_xlabel('dX') 
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
    plt.savefig(join(FIG_PATH, regions, 'Histogram of cluster positions' + SaveStr))
    print('Figures saved')
    
    
    
    
    
"""
@author: Marsa Taheri, extended from Sebastian's code

"""

from one.api import ONE
import numpy as np
from brainbox.io.one import load_spike_sorting_with_channel
from brainbox.plot import peri_event_time_histogram
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import brainbox as bb
from reproducible_ephys_functions import query

# %% Functions:
    
def cluster_peths_FR_FF_sliding(ts, align_times, pre_time=0.2, post_time=0.5, 
                                hist_win=0.1, N_SlidesPerWind = 5):
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
    :return: FR_sorted, FR_STD_sorted, FF_sorted, TimeVect_sorted
    :rtype: np.array
    """
    #slidingStep = bin_size/n_slides
    epoch = [-pre_time, post_time]
    tshift = hist_win/N_SlidesPerWind 
    
    FR_unsort, FRstd_unsort, FF_unsort, TimeVect = [], [], [], []
    for s in range(N_SlidesPerWind):
        ts_shift = ts[ts>(s*tshift)]
        PethsPerShift, BinnedSpPerShift = bb.singlecell.calculate_peths(ts_shift, np.ones_like(ts_shift), np.array([1]),
                                                                        (align_times + s*tshift), pre_time=abs(epoch[0]), 
                                                                        post_time=(epoch[1]- s*tshift), bin_size=hist_win,
                                                                        smoothing=0, return_fr=False) 
        # To Do: The post_time=(epoch[1]- s*tshift) might need to become post_time=(epoch[1]- s*hist_win) or smth similar!
        
        CountPerBinPerShift = BinnedSpPerShift.reshape(BinnedSpPerShift.shape[0], BinnedSpPerShift.shape[2])
        #FR_PerTrialPerShift = CountPerBinPerShift/hist_win
        FR_TrialAvgPerShift = np.nanmean(CountPerBinPerShift, axis=0)/hist_win
        FR_TrialSTDPerShift = np.nanstd(CountPerBinPerShift, axis=0)/hist_win #stdev of firing rate; same as np.std(CountPerBinPerShift/hist_win, axis=0)
        FF_PerShift = np.nanvar(CountPerBinPerShift, axis=0)/np.nanmean(CountPerBinPerShift, axis=0)
        TimeVect_PerShift = PethsPerShift['tscale'] + s*tshift #np.arange(FR_PerShift.size) * hist_win + tshift*s #per slide
    
        # Append this shifted result with previous ones:
        FR_unsort = np.hstack((FR_unsort, FR_TrialAvgPerShift))
        FRstd_unsort = np.hstack((FRstd_unsort, FR_TrialSTDPerShift))
        FF_unsort = np.hstack((FF_unsort, FF_PerShift))
        TimeVect = np.hstack((TimeVect, TimeVect_PerShift)) #stacks the time vectors

    #Sort the time and FR vectors:
    TimeVect_sorted = np.sort(TimeVect)
    FR_sorted = np.array([x for _,x in sorted(zip(TimeVect, FR_unsort))])
    FR_STD_sorted = np.array([x for _,x in sorted(zip(TimeVect, FRstd_unsort))])
    FF_sorted = np.array([x for _,x in sorted(zip(TimeVect, FF_unsort))])
    
    return FR_sorted, FR_STD_sorted, FF_sorted, TimeVect_sorted



# %% Run code:

# Identify the RS subject and neuron:
ClusterID = 398 #328 #180 #335 
names = ['ZFM-01592']# ['ibl_witten_29']#['DY_018'] #['DY_018']
BrainRegion = 'LP' # 'VIS'

# Time course details for plotting:
pre_time, post_time = 0.4, 0.8 #0.1, 0.35
Side = 'Left Stim' #'Right Stim' or 'Left Stim'
CorrChoice = 1 #Only include correct choices, incorrect ones, or all? 1 for Yes, 0 for including all choices, -1 for incorrect choices
AlignTo = 'Stim' #Align the trials to which, 'Movement' or 'Stim'?
constrasts_all = [1, 0.25, 0] #The constrasts to examine. Full list: [1., 0.25, 0.125, 0.0625, 0.]

# For using a sliding window:
SlidingWind = 1 # Use a sliding window for the plots? 1 for yes, 0 for no.
SlideBinSize = 0.06
SlideN = 3

# For saving figures:
NameStr = 'AlignTo' + AlignTo + '_Choice' + str(CorrChoice) + '_Slide' + str(SlidingWind) #Any string for the filename when saving the file
saveFig = 0 #At the end, save figure or not? 1 for yes.

one = ONE()
traj = query(behavior=True)
boundary_width = 0.01
base_grey = 0.15
fs = 18


for count, t in enumerate(traj):
    eid = t['session']['id']
    probe = t['probe_name']
    if t['session']['subject'] not in names:
        continue

    # load data
    try:
        spikes, clusters, channels = pickle.load(open("../data/data_{}_sorting_1.p".format(eid), "rb"))
    except FileNotFoundError:
        try:
            spk, clus, chn = load_spike_sorting_with_channel(eid, one=one)
            spikes, clusters, channels = spk[probe], clus[probe], chn[probe]
            pickle.dump((spikes, clusters, channels), (open("../data/data_{}.p".format(eid), "wb")))
        except KeyError:
            print(eid)
            continue


    contrasts = one.load_object(eid, 'trials',  attribute=['contrastLeft', 'contrastRight'])
    contrast_L, contrast_R = contrasts['contrastLeft'], contrasts['contrastRight']
    if AlignTo == 'Movement': #To align to movement times:
        times = one.load_dataset(eid, '_ibl_trials.firstMovement_times.npy') 
        timesStimOn = one.load_object(eid, 'trials', attribute=['stimOn_times'])['stimOn_times']
        StimWithChoice = np.logical_and(~np.isnan(timesStimOn), ~np.isnan(times)) #all the indices where there was both a stim and a choice time
        times = times[StimWithChoice]
        contrast_L, contrast_R = contrast_L[StimWithChoice], contrast_R[StimWithChoice] #Only keep those where there was both a stim and a choice
        
    elif AlignTo == 'Stim': #To align to stimulus On times:
        times = one.load_object(eid, 'trials', attribute=['stimOn_times'])['stimOn_times']
        contrast_L, contrast_R = contrast_L[~np.isnan(times)], contrast_R[~np.isnan(times)]
        times = times[~np.isnan(times)]
    
    if CorrChoice==1: #Also consider whether the choice was correct or not
        choice = one.load_dataset(eid, '_ibl_trials.choice.npy') 
        if AlignTo == 'Movement':
            choice = choice[StimWithChoice] #Only keep those where there was both a stim and a choice
        event_times_Lcorr = times[np.logical_and(np.logical_or(contrast_L >= 0, contrast_R == 0.), choice== 1)] #Stim L (any, even 0) and choice L/CW
        event_times_Rcorr = times[np.logical_and(np.logical_or(contrast_R >= 0, contrast_L == 0.), choice== -1)] #Stim R (any, even 0) and choice R/CCW
        #Choose which of these to consider for left/right stim:
        event_times_right = event_times_Rcorr
        event_times_left = event_times_Lcorr
 
        left_contrasts = contrast_L[np.logical_and(np.logical_or(contrast_L >= 0, contrast_R == 0.), choice== 1)]
        left_contrasts[np.isnan(left_contrasts)] = 0.
        right_contrasts = contrast_R[np.logical_and(np.logical_or(contrast_R >= 0, contrast_L == 0.), choice== -1)]
        right_contrasts[np.isnan(right_contrasts)] = 0.


    elif CorrChoice==-1: #include only incorrect choices
        choice = one.load_dataset(eid, '_ibl_trials.choice.npy') 
        if AlignTo == 'Movement':
            choice = choice[StimWithChoice] #Only keep those where there was both a stim and a choice
        event_times_Lincorr = times[np.logical_and(np.logical_or(contrast_L >= 0, contrast_R == 0.), choice== -1)] #Stim L (any, even 0) but incorr. choice R/CCW
        event_times_Rincorr = times[np.logical_and(np.logical_or(contrast_R >= 0, contrast_L == 0.), choice== 1)] #Stim R (any, even 0) but incorr. choice L/CW
        #Choose which of these to consider for left/right stim:
        event_times_right = event_times_Rincorr
        event_times_left = event_times_Lincorr

        left_contrasts = contrast_L[np.logical_and(np.logical_or(contrast_L >= 0, contrast_R == 0.), choice== -1)]
        left_contrasts[np.isnan(left_contrasts)] = 0.
        right_contrasts = contrast_R[np.logical_and(np.logical_or(contrast_R >= 0, contrast_L == 0.), choice== 1)]
        right_contrasts[np.isnan(right_contrasts)] = 0.

        
    elif CorrChoice==0: #include all choices
        event_times_left = times[np.logical_or(contrast_L >= 0, contrast_R == 0.)]  # !!! I changed this to greater OR EQUAL;
        event_times_right = times[np.logical_or(contrast_R >= 0, contrast_L == 0.)]  

        left_contrasts = contrast_L[np.logical_or(contrast_L >= 0, contrast_R == 0.)]
        left_contrasts[np.isnan(left_contrasts)] = 0.
        right_contrasts = contrast_R[np.logical_or(contrast_R >= 0, contrast_L == 0.)]
        right_contrasts[np.isnan(right_contrasts)] = 0. #MT: Can we distinguish between no stim. and 0 stim on one side??        


    
    if Side=='Right Stim':
        event_times_Side = event_times_right
    elif Side=='Left Stim':
        event_times_Side = event_times_left

    cluster_regions = channels.acronym[clusters.channels]
    neurons = np.where(np.logical_and(np.chararray.startswith(cluster_regions.astype('U9'), BrainRegion), clusters['metrics']['label'] == 1))[0]

    for j, neuron in enumerate(neurons):
        print("Warning, code is limited")
        if neuron != ClusterID:
            continue

        plt.figure(figsize=(9, 12)) 

        plt.subplot(3, 1, 1)
        counter = 0
        contrast_count_list = [0]
        for c in constrasts_all:
            if Side=='Right Stim':
                temp = right_contrasts == c
                
            elif Side=='Left Stim':    
                temp = left_contrasts == c

            print("{}, count {}".format(c, np.sum(temp)))

            clu_spks = spikes.times[spikes.clusters == neuron]
            for i, time in enumerate(event_times_Side[temp]):
                idx = np.bitwise_and(clu_spks >= time - pre_time, clu_spks <= time + post_time)
                event_spks = clu_spks[idx]
                plt.vlines(event_spks - time, counter - i, counter - i - 1)
            counter -= np.sum(temp)
            contrast_count_list.append(counter)
        ylabel_pos = []
        for i, c in enumerate(constrasts_all):
            top = contrast_count_list[i]
            bottom = contrast_count_list[i + 1]
            plt.fill_between([-pre_time, -pre_time + boundary_width], [top, top], [bottom, bottom],
                             zorder=3, color=str(1 - (base_grey + c * (1 - base_grey))))
            ylabel_pos.append((top - bottom) / 2 + bottom)

        plt.yticks(ylabel_pos, constrasts_all, size=fs)
        plt.axvline(0, color='k', ls='--')
        plt.xlim(left=-pre_time, right=post_time)
        plt.ylim(top=0, bottom=counter)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.tick_params(left=False, right=False, labelbottom=False, bottom=False)
        plt.title("Contrast: {}, Choice Corr {}, Aligned to {}".format(Side, CorrChoice, AlignTo), loc='left', size=fs+2) 

        
        for c in constrasts_all:
            if Side=='Right Stim':
                mask = right_contrasts == c
            elif Side=='Left Stim':    
                mask = left_contrasts == c
            
            if sum(mask)==0:
                print('No events for side %s, contrast %d, and choice %d' %(Side, c, CorrChoice))
                continue
            
            
            if SlidingWind==0:
                psths, BinnedSpikes = bb.singlecell.calculate_peths(spikes.times, spikes.clusters, neurons, event_times_Side[mask],
                                                                    pre_time=pre_time, post_time=post_time, smoothing=0, bin_size=0.04)

                plt.subplot(3, 1, 2)
                plt.plot(psths.tscale, psths.means[j], c=str(1 - (base_grey + c * (1 - base_grey))))
                plt.fill_between(psths.tscale,
                                 psths.means[j] + psths.stds[j] / np.sqrt(np.sum(mask)),
                                 psths.means[j] - psths.stds[j] / np.sqrt(np.sum(mask)),
                                 color=str(1 - (base_grey + c * (1 - base_grey))), alpha=0.3)
            
                plt.subplot(3, 1, 3)
                #Calculate Fano Factor:
                CountPerBin=BinnedSpikes[:, j, :]
                FanoFactor = np.nanvar(CountPerBin, axis=0)/np.nanmean(CountPerBin, axis=0)
                plt.plot(psths.tscale, FanoFactor, c=str(1 - (base_grey + c * (1 - base_grey))))
           
            elif SlidingWind==1:
                #Use sliding window to get FR and FF over time; 1 neuron at a time so no neurons[j] or spikes.clusters not needed
                FR_slide, FR_STD_slide, FF_slide, TimeVect_slide = cluster_peths_FR_FF_sliding(spikes.times[spikes.clusters == neurons[j]], 
                                                                                               event_times_Side[mask],
                                                                                               pre_time=pre_time, post_time=post_time,
                                                                                               hist_win=SlideBinSize, N_SlidesPerWind = SlideN) 
                plt.subplot(3, 1, 2)
                plt.plot(TimeVect_slide, FR_slide, c=str(1 - (base_grey + c * (1 - base_grey))))
                plt.fill_between(TimeVect_slide,
                                 FR_slide + FR_STD_slide / np.sqrt(np.sum(mask)),
                                 FR_slide - FR_STD_slide / np.sqrt(np.sum(mask)),
                                 color=str(1 - (base_grey + c * (1 - base_grey))), alpha=0.3)
            
                plt.subplot(3, 1, 3)
                plt.plot(TimeVect_slide, FF_slide, c=str(1 - (base_grey + c * (1 - base_grey))))                
                
            
        plt.subplot(3, 1, 2)
        plt.axvline(0, color='k', ls='--')
        plt.xlim(left=-pre_time, right=post_time)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        # plt.yticks([0, 25, 50, 75], [0, 25, 50, 75], size=fs)
        plt.yticks(size=fs)
        plt.xticks([-0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], 
                   [-0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], size=fs)
        plt.ylabel("Firing rate (sp/s)", size=fs+3)
        #plt.xlabel("Time from stim onset (s)", size=fs+3)

        plt.subplot(3, 1, 3)
        plt.axvline(0, color='k', ls='--')
        plt.xlim(left=-pre_time, right=post_time)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.yticks(size=fs)
        plt.xticks([-0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], 
                   [-0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], size=fs)
        plt.ylabel("Fano Factor", size=fs+3)
        plt.xlabel("Time from stim onset (s)", size=fs+3)
        
        
        if saveFig==1:
            plt.savefig("SaveMyFigures/{}, {}, {}, {}".format(t['session']['subject'], neuron, Side, NameStr)) #MT modified to include folder name and Side
            #plt.close()

        plt.show()
        plt.close()
        
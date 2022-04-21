"""
@author: Marsa Taheri, extended from Sebastian's code

#NEED TO UPDATE TO DOWNLOAD PYKILOSORT VS KS2 DATA, DEPENDING ON RECORDING!
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
from brainbox.metrics.single_units import spike_sorting_metrics, quick_unit_metrics
import pandas as pd

from MT_selectivity import p_value_analysis


from brainbox.task.trials import find_trial_ids

def plot_raster_and_psth(pid, neuron, align_event, contrasts, side, feedback, one, ba):


    eid, probe = one.pid2eid(pid)
    sl = SpikeSortingLoader(eid=eid, pname=probe, one=one, atlas=ba)
    spikes, clusters, channels = sl.load_spike_sorting()
    clusters = sl.merge_clusters(spikes, clusters, channels)

    spike_idx = np.isin(spikes['clusters'], neuron)
    if np.sum(spike_idx) == 0:
        print('warning, warning')

    trials = one.load_object(eid, 'trials', collection='alf')

    trial_id, _ = find_trial_ids(trials, side=side, choice=feedback, contrasts=contrasts)








    # %% Run code:

    # Identify the RS subject and neuron:
    ClusterID = 614 #345 #147 #289 #241 #335#335#180 #406 #335  #180 #398 #328 #
    names = ['SWC_054']# ['NYU-12'] #['ibl_witten_27'] #['SWC_054']# ['DY_018'] #['ZFM-01592'] #['ibl_witten_29'] #['DY_018']
    BrainRegion = 'VIS' #'VIS' #'VIS' #'LP' #'VIS'#'VIS' #'LP'#'VIS' #'LP' #

    # Time course details for plotting:
    pre_time, post_time = 0.4, 0.6 #0.4, 0.8 #0.1, 0.35
    Side = 'Right Stim' #'Right Stim' or 'Left Stim'
    CorrChoice = 'Correct' #Only include correct choices, incorrect ones, or all? Options: 'Correct', 'Incorr', 'All Choices'
    AlignTo = 'Movement' #Align the trials to which, 'Movement' or 'Stim'?
    constrasts_all = [1., 0.25, 0.125, 0.0625, 0.] #[1., 0.] #The constrasts to examine. Full list: [1., 0.25, 0.125, 0.0625, 0.]

    # For using a sliding window:
    SlidingWind = 1 # Use a sliding window for the plots? 1 for yes, 0 for no.
    SlideBinSize = 0.06#0.04
    SlideN = 3#2
    CausalOrNot = 1 #0 or 1; 1 means it's causal, i.e., time points placed at the end of each hist_win such that only past events are taken into account.

    # For saving figures:
    NameStr = 'AlignTo' + AlignTo + '_' +CorrChoice + '_SlideCausal' + str(SlidingWind) #Any string for the filename when saving the file
    saveFig = 0 #At the end, save figure or not? 1 for yes.

    one = ONE()
    traj = query(behavior=True)
    boundary_width = 0.01
    base_grey = 0.3 #0.15
    fs = 18 #figure size


    for count, t in enumerate(traj):
        eid = t['session']['id']
        probe = t['probe_name']
        if t['session']['subject'] not in names:
            continue


        # #Selectivity:
        # Ps = p_value_analysis(eid, probe)

        # load data
        try:
            # #Try Pykilosort:
            spikes = one.load_object(eid, 'spikes', collection='alf/{}/pykilosort'.format(probe))
            clusters = one.load_object(eid, 'clusters', collection='alf/{}/pykilosort'.format(probe))
            channels = bb.io.one.load_channel_locations(eid, one=one, probe=probe, aligned=True)
            clusters = bb.io.one.merge_clusters_channels(dic_clus={probe: clusters}, channels=channels)[probe]
            channels = channels[probe]
            #spikes, clusters, channels = pickle.load(open("../data/data_{}_sorting_1.p".format(eid), "rb"))

        # except FileNotFoundError:
        #     try:
        #         # spk, clus, chn = load_spike_sorting_with_channel(eid, one=one)
        #         # spikes, clusters, channels = spk[probe], clus[probe], chn[probe]

        #         #Try ks2:
        #         spikes = one.load_object(eid, 'spikes', collection='alf/{}'.format(probe), revision='')
        #         clusters = one.load_object(eid, 'clusters', collection='alf/{}'.format(probe), revision='')
        #         channels = bb.io.one.load_channel_locations(eid, one=one, probe=probe, aligned=True)
        #         clusters = bb.io.one.merge_clusters_channels(dic_clus={probe: clusters}, channels=channels)[probe]
        #         channels = channels[probe]

        #         # # #Try Pykilosort:
        #         # spikes = one.load_object(eid, 'spikes', collection='alf/{}/pykilosort'.format(probe))
        #         # clusters = one.load_object(eid, 'clusters', collection='alf/{}/pykilosort'.format(probe))
        #         # channels = bb.io.one.load_channel_locations(eid, one=one, probe=probe, aligned=True)
        #         # clusters = bb.io.one.merge_clusters_channels(dic_clus={probe: clusters}, channels=channels)[probe]
        #         # channels = channels[probe]

        #         pickle.dump((spikes, clusters, channels), (open("../data/data_{}.p".format(eid), "wb")))

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

        if CorrChoice=='Correct': #Also consider whether the choice was correct or not
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


        elif CorrChoice=='Incorr': #include only incorrect choices
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


        elif CorrChoice=='All Choices': #include all choices
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


        try:
            cluster_regions = channels['acronym'][clusters.channels] #channels.acronym[clusters.channels]
            neurons = np.where(np.logical_and(np.chararray.startswith(cluster_regions.astype('U9'), BrainRegion), clusters['metrics']['label'] == 1))[0]
        except:
            #When using Pykilosort and no 'metrics' is found:
            # #Sometimes 'channels' using load_spike_sorting_with_channel does not have 'amps' needed to get 'metrics'; therefore use this method now:
            # spikes = one.load_object(eid, 'spikes', collection='alf/{}/pykilosort'.format(probe))
            # clusters = one.load_object(eid, 'clusters', collection='alf/{}/pykilosort'.format(probe))
            # channels = bb.io.one.load_channel_locations(eid, one=one, probe=probe, aligned=True)
            # clusters = bb.io.one.merge_clusters_channels(dic_clus={probe: clusters}, channels=channels)[probe]
            # channels = channels[probe]

            r = quick_unit_metrics(spikes.clusters, spikes.times, spikes.amps, spikes.depths, cluster_ids=np.arange(clusters.channels.size)) #Need to convert to a dataframe
            dict_metrics = {} #converting r (a util.Bunch object that's really a dictionary) to a regular dictionary
            for key in r.keys():
                dict_metrics[key] = r[key]
            df_metrics = pd.DataFrame.from_dict(dict_metrics) #converting dictionary to a Pandas dataframe
            clusters['metrics'] = df_metrics


            pickle.dump((spikes, clusters, channels), (open("../data/data_{}.p".format(eid), "wb")))

            cluster_regions = channels['acronym'][clusters.channels] #channels.acronym[clusters.channels]
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
                    plt.vlines(event_spks - time, counter - i, counter - i - 1, color='k')
                counter -= np.sum(temp)
                contrast_count_list.append(counter)
            ylabel_pos = []
            for i, c in enumerate(constrasts_all):
                top = contrast_count_list[i]
                bottom = contrast_count_list[i + 1]
                #Position of the contrast colorbar:
                plt.fill_between([-pre_time - SlideBinSize/2, -pre_time - SlideBinSize/2 + boundary_width],
                                 [top, top], [bottom, bottom],
                                 zorder=3, color=str(1 - (base_grey + c * (1 - base_grey))))
                ylabel_pos.append((top - bottom) / 2 + bottom)

            plt.yticks(ylabel_pos, constrasts_all, size=fs)
            if AlignTo == 'Stim':
                plt.axvline(0, color=(0, 0.5, 1), ls='--', linewidth=2)
            elif AlignTo == 'Movement':
                plt.axvline(0, color='g', ls='--', linewidth=2)
            plt.xlim(left= -pre_time - SlideBinSize/2, right= post_time + SlideBinSize/2) #(left=-pre_time, right=post_time)
            plt.ylim(top=0, bottom=counter)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.gca().spines['bottom'].set_visible(False)
            plt.tick_params(left=False, right=False, labelbottom=False, bottom=False)
            plt.title("Contrast    {}, {}, Aligned to {}".format(Side, CorrChoice, AlignTo), loc='left', size=fs)
            #plt.title("Contrast ({})".format(Side), loc='left', size=fs+2)


            for c in constrasts_all:
                if Side=='Right Stim':
                    mask = right_contrasts == c
                elif Side=='Left Stim':
                    mask = left_contrasts == c

                if sum(mask)==0:
                    print('No events for side %s, contrast %d, and choice %s' %(Side, c, CorrChoice))
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
                #Use sliding window to get FR and FF over time; 1 neuron at a time so neurons[j] or spikes.clusters are not needed
                FR_slide, FR_STD_slide, FF_slide, TimeVect_slide = cluster_peths_FR_FF_sliding(spikes.times[spikes.clusters == neurons[j]], 
                                                                                               event_times_Side[mask],
                                                                                               pre_time=pre_time, post_time=post_time,
                                                                                               hist_win=SlideBinSize, N_SlidesPerWind = SlideN,
                                                                                               causal=CausalOrNot) 
                plt.subplot(3, 1, 2)
                plt.plot(TimeVect_slide, FR_slide, c=str(1 - (base_grey + c * (1 - base_grey))))
                plt.fill_between(TimeVect_slide,
                                 FR_slide + FR_STD_slide / np.sqrt(np.sum(mask)),
                                 FR_slide - FR_STD_slide / np.sqrt(np.sum(mask)),
                                 color=str(1 - (base_grey + c * (1 - base_grey))), alpha=0.3)
            
                plt.subplot(3, 1, 3)
                plt.plot(TimeVect_slide, FF_slide, c=str(1 - (base_grey + c * (1 - base_grey))))                
                
            
        plt.subplot(3, 1, 2)
        if AlignTo == 'Stim':
            plt.axvline(0, color=(0, 0.5, 1), ls='--', linewidth=2)
            #plt.xlabel("Time from stim onset (s)", size=fs+3)  
        elif AlignTo == 'Movement':
            plt.axvline(0, color='g', ls='--', linewidth=2)
            #plt.xlabel("Time from movement onset (s)", size=fs+3)         
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        # plt.yticks([0, 25, 50, 75], [0, 25, 50, 75], size=fs)
        plt.yticks(size=fs)
        plt.xticks([-0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], 
                   [-0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], size=fs)
        plt.ylabel("Firing rate (sp/s)", size=fs+3)
        plt.xlim(left=-pre_time, right=post_time)
        #plt.xlim(left= -pre_time - SlideBinSize/2, right= post_time + SlideBinSize/2)
        

        plt.subplot(3, 1, 3)
        plt.xlim(left=-pre_time, right=post_time)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.yticks(size=fs)
        plt.xticks([-0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], 
                   [-0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], size=fs)
        plt.ylabel("Fano Factor", size=fs+3)
        if AlignTo == 'Stim':
            plt.axvline(0, color=(0, 0.5, 1), ls='--', linewidth=2)
            plt.xlabel("Time from stim onset (s)", size=fs+3)            
        elif AlignTo == 'Movement':
            plt.axvline(0, color='g', ls='--', linewidth=2)
            plt.xlabel("Time from movement onset (s)", size=fs+3)            
        #plt.xlim(left= -pre_time - SlideBinSize/2, right= post_time + SlideBinSize/2)
        plt.xlim(left= -pre_time, right= post_time)
        
        if saveFig==1:
            plt.savefig("SaveMyFigures/{}, {}, {}, {}".format(t['session']['subject'], neuron, Side, NameStr)) #MT modified to include folder name and Side
            #plt.close()

        plt.show()
        plt.close()
        
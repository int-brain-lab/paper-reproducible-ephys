
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
from reproducible_ephys_functions import query, labs
from reproducible_ephys_paths import FIG_PATH
from oneibl.one import ONE
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.tri as mtri

from pylab import *
import scipy.io as sio       
from brainbox.singlecell import calculate_peths
from brainbox.plot import peri_event_time_histogram
  
from ibllib.atlas import atlas, AllenAtlas
from ibllib.pipes import histology

one = ONE()
ba = AllenAtlas()

# Query repeated site trajectories
#traj = query()
<<<<<<< Updated upstream
traj = query(behavior=True)

fr_all = []
FF_all = []
MaxFR_all=[]
xvals_all=[]
yvals_all=[]
zvals_all=[]
LabNum_all = []
amps_all=[]
ptt_all=[]
PeriEventMeanFR_all0, FFPeriEvent0 = [], []
PeriEventMeanFR_all1, FFPeriEvent1 = [], []
PeriEventMeanFR_all2, FFPeriEvent2 = [], []
PeriEventMeanFR_all3, FFPeriEvent3 = [], []
PeriEventMeanFR_all4, FFPeriEvent4 = [], []
PeriEventMeanFR_all5, FFPeriEvent5 = [], []
PeriEventMeanFR_all6, FFPeriEvent6 = [], []

#apply_baseline = False
# event = 'Stim'
# eventTypes = ['LCont','RCont','NoCont','LCont100','RCont100']

regions = ['LP']#, VISa', 'CA1', 'DG', 'LP', 'PO']

# %% Functions (modified from Seb's normalise_neurons function)
=======
#traj = query(behavior=True)
ListOfEIDs = eid_list()

# Initialize dataframe
ClusterFeatures = pd.DataFrame() #['eID', 'LabID','clusterID', 'AvgFR', 'AvgFF', 'amps', 'peak_trough', 'Xloc', 'Yloc', 'Zloc']
PeriEventFRandFF = pd.DataFrame() #['FR stimOn', 'FR '] FR and FF averaged over entire session

Ntrials=[] #N of behavioral trials

PeriEventMeanFR_all0, FFPeriEvent0_all = [], []
# PeriEventMeanFR_all1, FFPeriEvent1_all = [], []
# PeriEventMeanFR_all2, FFPeriEvent2_all = [], []
# PeriEventMeanFR_all3, FFPeriEvent3_all = [], []
# PeriEventMeanFR_all4, FFPeriEvent4_all = [], []
# PeriEventMeanFR_all5, FFPeriEvent5_all = [], []
# PeriEventMeanFR_all6, FFPeriEvent6_all = [], []

regions = 'PPC' #[PPC', 'CA1', 'DG', 'LP', 'PO']
SaveFigs=0 # Used to choose whether to save the figures at the end; to save, set equal to 1

    
# %% Functions (modified from others' codes)
# used if firing rates are normalized
>>>>>>> Stashed changes
def normalise_act(activities, spikes, probe, clusterIDs, base_line_times):
    activity_pre, _ = calculate_peths(spikes[probe]['times'], spikes[probe]['clusters'],
                                      np.array(clusterIDs), base_line_times,
                                      pre_time=0.4, post_time=-0.2, smoothing=0, bin_size=0.01) #-0.2 s means 200 ms BEFORE the stim.
    baseline = np.mean(activity_pre.means, axis=1)

    normed_activities = []
    for a in activities:
        normed_activities.append(((a.means.T - baseline) / (1 + baseline)).T)

    return normed_activities

# %% Loop through repeated site recordings and extract the data

for i in range(2):#len(traj)):
    print('Processing repeated site recording %d of %d' % (i+1, len(traj)))
<<<<<<< Updated upstream
    fr = []
    FF=[]
    MaxFR=[]
    xvals =[]
    yvals =[]
    zvals =[]
    LabNumArray = []
    PeriEventMeanFRperClust0, FFPeriEvent0_all = [], []
    PeriEventMeanFRperClust1, FFPeriEvent1_all = [], []
    PeriEventMeanFRperClust2, FFPeriEvent2_all = [], []
    PeriEventMeanFRperClust3, FFPeriEvent3_all = [], []
    PeriEventMeanFRperClust4, FFPeriEvent4_all = [], []
    PeriEventMeanFRperClust5, FFPeriEvent5_all = [], []
    PeriEventMeanFRperClust6, FFPeriEvent6_all = [], []
    #PeriEventMeanFRperClust = [0 for _ in eventTypes] #{x: [] for x in eventTypes}
=======
    fr, FF, LabNumArray = [], [], []
    xvals, yvals, zvals =[], [], []
    FRev0, FFev0 =[], []

    PeriEventMeanFRperClust0, FFPeriEvent0 = [], []
    # PeriEventMeanFRperClust1, FFPeriEvent1 = [], []
    # PeriEventMeanFRperClust2, FFPeriEvent2 = [], []
    # PeriEventMeanFRperClust3, FFPeriEvent3 = [], []
    # PeriEventMeanFRperClust4, FFPeriEvent4 = [], []
    # PeriEventMeanFRperClust5, FFPeriEvent5 = [], []
    # PeriEventMeanFRperClust6, FFPeriEvent6 = [], []
    ## PeriEventMeanFRperClust = [0 for _ in eventTypes] #{x: [] for x in eventTypes}
>>>>>>> Stashed changes
    
    
    # Load in data
    eid = traj[i]['session']['id']
    try:
<<<<<<< Updated upstream
        spikes, clusters, channels = bbone.load_spike_sorting_with_channel(eid, one=one)
=======
        ###spikes, clusters, channels = bbone.load_spike_sorting_with_channel(eid, one=one)
        #spikes = one.load_object(eid, 'spikes', collection='alf/{}'.format(probe), revision='ks2_preproc_tests')
        #clusters = one.load_object(eid, 'clusters', collection='alf/{}'.format(probe), revision='ks2_preproc_tests')
        spikes = one.load_object(eid, 'spikes', collection='alf/{}'.format(probe), revision='')
        clusters = one.load_object(eid, 'clusters', collection='alf/{}'.format(probe), revision='')
        channels = bbone.load_channel_locations(eid, one=one, probe=probe, aligned=True)
        clusters = bbone.merge_clusters_channels(dic_clus={probe: clusters}, channels=channels)[probe]
>>>>>>> Stashed changes
    except:
        continue

    # Get coordinates of micro-manipulator and histology
    hist = one.alyx.rest('trajectories', 'list', provenance='Histology track',
                     probe_insertion=traj[i]['probe_insertion'])
    manipulator = one.alyx.rest('trajectories', 'list', provenance='Micro-manipulator',
                                probe_insertion=traj[i]['probe_insertion'])
    if (len(hist) == 0) or (len(manipulator) == 0):
        continue


    # find probe name and lab name, and associate the lab name with a number:
    probe = traj[i]['probe_name']
    LabName = traj[i]['session']['lab']
    lab_number_map, _, _ = labs() #lab_number_map, institution_map, lab_colors = labs()
    LabNum = int(lab_number_map.get(LabName)[-1])
  
    # Get firing rates
    clusterIDs  = clusters[probe]['metrics']['cluster_id'][clusters[probe]['acronym'] == regions][clusters[probe]['metrics']['label'] == 1]
#DG
    # clusterIDs  = clusters[probe]['metrics']['cluster_id'][clusters[probe]['acronym'] == 'DG-mo'][clusters[probe]['metrics']['label'] == 1]
    # clusterIDs = clusterIDs.append(clusters[probe]['metrics']['cluster_id'][clusters[probe]['acronym'] == 'DG-po'])[clusters[probe]['metrics']['label'] == 1]
    # clusterIDs = clusterIDs.append(clusters[probe]['metrics']['cluster_id'][clusters[probe]['acronym'] == 'DG-sg'])[clusters[probe]['metrics']['label'] == 1]
   
        
    #Find cluster waveform (spike amp & width):
    amps = one.load_dataset(eid, 'clusters.amps.npy', collection='alf/{}'.format(probe))
    amps = amps[clusterIDs]
    ptt = one.load_dataset(eid, 'clusters.peakToTrough.npy', collection='alf/{}'.format(probe))
    ptt = ptt[clusterIDs]

    
    #Find specific task Event times:
#event == 'Stim' 
    timesStimOn = one.load(eid, dataset_types=['trials.stimOn_times'])[0]
    base_line_times = one.load(eid, dataset_types=['trials.stimOn_times'])[0]
    base_line_times = base_line_times[~np.isnan(base_line_times)]
    contrast_L, contrast_R = one.load(eid, dataset_types=['trials.contrastLeft', 'trials.contrastRight'])
    contrast_L, contrast_R = contrast_L[~np.isnan(timesStimOn)], contrast_R[~np.isnan(timesStimOn)]
    timesStimOn = timesStimOn[~np.isnan(timesStimOn)]
    event_times_left = timesStimOn[contrast_L > 0]
    event_times_right = timesStimOn[contrast_R > 0]
    event_times_0 = timesStimOn[np.logical_or(contrast_R == 0, contrast_L == 0)]
    event_times_left100 = timesStimOn[contrast_L == 1]
    event_times_right100 = timesStimOn[contrast_R == 1]
# #event == 'Block' and the stim is on the same side (make sure the L/R are correct)
    # block_prob = one.load(eid, dataset_types=['trials.probabilityLeft'])
    # event_times_LBlockStim = timesStimOn[(block_prob[0] == 0.8) & (contrast_L == 1)]
    # event_times_RBlockStim = timesStimOn[(block_prob[0] == 0.2) & (contrast_R == 1)]
#event == 'Move'
    times1stMove = one.load(eid, dataset_types=['trials.firstMovement_times'])[0]
    # base_line_times = one.load(eid, dataset_types=['trials.stimOn_times'])[0]
    # base_line_times = base_line_times[~np.isnan(base_line_times)]
    choice = one.load(eid, dataset_types=['trials.choice'])[0]
    if (~np.isnan(times1stMove)).sum() < 300:
        continue
    choice = choice[~np.isnan(times1stMove)]
    times1stMove = times1stMove[~np.isnan(times1stMove)]
    event_times_leftChoice = times1stMove[choice == -1]
    event_times_rightChoice = times1stMove[choice == 1]


    #All Events into one array:    
    event_times = [event_times_left, event_times_right, event_times_0, event_times_left100, 
                   event_times_right100, event_times_leftChoice, event_times_rightChoice]
    
    #if condition returns True, nothing happens; if condition returns False, AssertionError is raised:
    assert np.sum(np.isnan(event_times_left)) == 0
    assert np.sum(np.isnan(event_times_right)) == 0
    assert np.sum(np.isnan(event_times_leftChoice)) == 0
    assert np.sum(np.isnan(event_times_rightChoice)) == 0
    assert np.sum(np.isnan(base_line_times)) == 0
    
    
    activities = []
    BinnedSpikes = []
    count = 0
    for etime in event_times:
        #pre_time, post_time = 0.2, 2#0.4 #Hyun is looking at 0.2 s before and 2 s after the stimulus onset
        if count<5:
            pre_time, post_time = 0.2, 2#0.4 #Hyun is looking at 0.2 s before and 2 s after the stimulus onset
        elif count<7:
            pre_time, post_time = 0.4, 0.2
<<<<<<< Updated upstream
        a, b = calculate_peths(spikes[probe]['times'], spikes[probe]['clusters'], np.array(clusterIDs),
                               etime, pre_time=pre_time, post_time=post_time, smoothing=0, bin_size=0.01)
        activities.append(a)
        BinnedSpikes.append(b)
        count += 1
        
=======
        a, b = calculate_peths(spikes['times'], spikes['clusters'], np.array(clusterIDs),
                               etime, pre_time=pre_time, post_time=post_time, smoothing=0, bin_size=0.05)
        activities.append(a)
        BinnedSpikes.append(b)
        # if count == 0:
        #     peri_event_time_histogram(spikes['times'], spikes['clusters'], event_times[0], np.array(clusterIDs),  # Everything you need for a basic plot
        #                               t_before=pre_time, t_after=post_time, bin_size=0.025, smoothing=0, as_rate=True,
        #                               include_raster=True, n_rasters=4)
        count += 1        
        
            
>>>>>>> Stashed changes
    # activities_NotNorm = [a.means for a in activities]
    # activities_Normalized = normalise_act(activities,  spikes, probe, clusterIDs, base_line_times)
    #activities = activities_Normalized
    #After the part above, activities[0].means will be equivalent to activities_Normalized[0] or activities_NotNorm[0]
    
    for k, cluster in enumerate(clusterIDs):
<<<<<<< Updated upstream
        fr.append(spikes[probe]['times'][spikes[probe]['clusters'] == cluster].shape[0]/ spikes[probe]['times'][-1])
=======
        fr.append(spikes['times'][spikes['clusters'] == cluster].shape[0]/ spikes['times'][-1])
        Ntrials.append(len(event_times_left)+len(event_times_right) + len(event_times_0)) #Total # of behav. trials per eID (shows as per cluster)
>>>>>>> Stashed changes
        
        #This part calculates the f.r. in bins of 50 ms, then calculates the variance and Fano Factor:
        SpikesOfCluster = spikes[probe]['times'][spikes[probe]['clusters'] == cluster]
        time_bins = arange(0, spikes[probe]['times'][-1], 0.05)
        SpkIncrements50ms, _ = histogram(SpikesOfCluster, time_bins) #vector containing the # of spikes/counts in each 50 ms increment.

        FF50ms = SpkIncrements50ms.var()/SpkIncrements50ms.mean()
        FF.append(FF50ms)        
        # MeanFR = np.mean(SpkIncrements50ms)*20 #avg spikes in 50 ms, converted to per second (result same as fr.append(...) calculation)
           
        LabNumArray.append(LabNum)
    
        xvals.append(clusters[probe]['x'][clusters[probe]['metrics']['cluster_id'] == cluster])
        yvals.append(clusters[probe]['y'][clusters[probe]['metrics']['cluster_id'] == cluster])
        zvals.append(clusters[probe]['z'][clusters[probe]['metrics']['cluster_id'] == cluster])
        
        #Now find the average firing rate around the event only, for each cluster:
        FRev0.append(mean(activities[0].means[activities[0].cscale == cluster]))
            
        PeriEventMeanFRperClust0.append(mean(activities[0].means[activities[0].cscale == cluster]))

        #Now find the Fano Factor around the event only, for each cluster:
        MeanFRoverT = activities[0].means[activities[0].cscale == cluster] #mean for 1 cluster over trials for each binned time
        VarFRoverT = square(activities[0].stds[activities[0].cscale == cluster]) 
        FFoverT = VarFRoverT/MeanFRoverT
        #FFev0.append(var(MeanFRoverT)/mean(MeanFRoverT)) #Old method
        FFev0.append(list(min(np.transpose(FFoverT))))#(FFoverT[2])
        #The following 2 lines are equivalent:
        #FFev0.append(var(MeanFRoverT)/mean(MeanFRoverT))
        #FFev0.append(var(activities[0].means[activities[0].cscale == cluster])/mean(activities[0].means[activities[0].cscale == cluster]))

        FFPeriEvent0.append(var(activities[0].means[activities[0].cscale == cluster])/mean(activities[0].means[activities[0].cscale == cluster]))
        # FFPeriEvent1.append(var(activities[1].means[activities[1].cscale == cluster])/mean(activities[1].means[activities[1].cscale == cluster]))
        # FFPeriEvent2.append(var(activities[2].means[activities[2].cscale == cluster])/mean(activities[2].means[activities[2].cscale == cluster]))
        # FFPeriEvent3.append(var(activities[3].means[activities[3].cscale == cluster])/mean(activities[3].means[activities[3].cscale == cluster]))
        # FFPeriEvent4.append(var(activities[4].means[activities[4].cscale == cluster])/mean(activities[4].means[activities[4].cscale == cluster]))
        # FFPeriEvent5.append(var(activities[5].means[activities[5].cscale == cluster])/mean(activities[5].means[activities[5].cscale == cluster]))
        # FFPeriEvent6.append(var(activities[6].means[activities[6].cscale == cluster])/mean(activities[6].means[activities[6].cscale == cluster]))
        
    
    #Save all the data:
<<<<<<< Updated upstream
    fr_all.append(np.array(fr))
    FF_all.append(np.array(FF))
    MaxFR_all.append(np.array(MaxFR))
    LabNum_all.append(LabNumArray)
    amps_all.append(amps)
    ptt_all.append(ptt)
    xvals_all.append(xvals)
    yvals_all.append(yvals)
    zvals_all.append(zvals)
=======
    if len(clusterIDs)>1:
        data1 = np.array([np.repeat(eid, len(clusterIDs)), LabNumArray, np.array(clusterIDs), np.array(fr), np.array(FF),
                          amps1[clusterIDs], ptt1[clusterIDs], np.squeeze(xvals), np.squeeze(yvals), np.squeeze(zvals)])
        data1 = np.transpose(data1); #data1.reshape(len(clusterIDs),len(columns))
        
        #data frame for peri-event calculations:
        dataPeri = np.array([np.repeat(eid, len(clusterIDs)), LabNumArray, np.array(clusterIDs), np.array(Ntrials),
                             np.array(FRev0), np.array(FFev0)])
        dataPeri = np.transpose(dataPeri)

    elif len(clusterIDs)==1:
        data1 = np.array([eid, np.squeeze(LabNumArray), np.squeeze(np.array(clusterIDs)), np.squeeze(np.array(fr)), np.squeeze(np.array(FF)),
                          np.squeeze(amps1[clusterIDs]), np.squeeze(ptt1[clusterIDs]), np.squeeze(xvals), np.squeeze(yvals), np.squeeze(zvals)])
        data1 = data1.reshape(len(clusterIDs),10)

        #data frame for peri-event calculations:
        dataPeri = np.array([eid, np.squeeze(LabNumArray), np.squeeze(np.array(clusterIDs)), np.squeeze(np.array(Ntrials)),
                             np.squeeze(np.array(FRev0)), np.squeeze(np.array(FFev0))])
        dataPeri = dataPeri.reshape(len(clusterIDs),10)


    #Data frame for session-averaged calculations:
    columns=['eID', 'LabID','clusterID', 'AvgFR', 'AvgFF', 'amps', 'peak_trough', 'Xloc', 'Yloc', 'Zloc']
    df = pd.DataFrame(data1, columns=columns,
                      index=np.arange(len(ClusterFeatures), len(ClusterFeatures)+len(clusterIDs))) #index=np.arange(0,len(clusterIDs)))
    ClusterFeatures = ClusterFeatures.append(df) 
    
    #Data frame for perievent calculations:
    columnsPeri=['eID', 'LabID','clusterID', 'Ntrial', 'FR_event0', 'FF_event0']
    dfPeri = pd.DataFrame(dataPeri, columns=columnsPeri,
                      index=np.arange(len(ClusterFeatures), len(ClusterFeatures)+len(clusterIDs))) #index=np.arange(0,len(clusterIDs)))
    PeriEventFRandFF = PeriEventFRandFF.append(dfPeri)
    
    
>>>>>>> Stashed changes
    
    PeriEventMeanFR_all0.append(np.array(PeriEventMeanFRperClust0))
    FFPeriEvent0_all.append(np.array(FFPeriEvent0))
    # FFPeriEvent1_all.append(np.array(FFPeriEvent1))
    # FFPeriEvent2_all.append(np.array(FFPeriEvent2))
    # FFPeriEvent3_all.append(np.array(FFPeriEvent3))
    # FFPeriEvent4_all.append(np.array(FFPeriEvent4))
    # FFPeriEvent5_all.append(np.array(FFPeriEvent5))
    # FFPeriEvent6_all.append(np.array(FFPeriEvent6))


FR_stacked = np.hstack(fr_all)
FF_stacked = np.hstack(FF_all)
MaxFR50ms_stacked = np.hstack(MaxFR_all)
xvals_all_filtered = [x for x in xvals_all if x] #removes empty lists from list
X_stacked = np.vstack(xvals_all_filtered)
yvals_all_filtered = [x for x in yvals_all if x]
Y_stacked = np.vstack(yvals_all_filtered)
zvals_all_filtered = [x for x in zvals_all if x]
Z_stacked = np.vstack(zvals_all_filtered)
LabNum_filtered = [x for x in LabNum_all if x]
LabNum_stacked = np.hstack(LabNum_filtered)
amps_stacked = np.hstack(amps_all)
ptt_stacked = np.hstack(ptt_all)

PeriEventFR_stacked0 = np.reshape(np.hstack(PeriEventMeanFR_all0), (len(np.hstack(PeriEventMeanFR_all0)),1))
PeriEventFF_stacked0 = np.reshape(np.hstack(FFPeriEvent0_all), (len(np.hstack(FFPeriEvent0_all)),1))
# PeriEventFF_stacked1 = np.reshape(np.hstack(FFPeriEvent1_all), (len(np.hstack(FFPeriEvent1_all)),1))
# PeriEventFF_stacked2 = np.reshape(np.hstack(FFPeriEvent2_all), (len(np.hstack(FFPeriEvent2_all)),1))
# PeriEventFF_stacked3 = np.reshape(np.hstack(FFPeriEvent3_all), (len(np.hstack(FFPeriEvent3_all)),1))
# PeriEventFF_stacked4 = np.reshape(np.hstack(FFPeriEvent4_all), (len(np.hstack(FFPeriEvent4_all)),1))
# PeriEventFF_stacked5 = np.reshape(np.hstack(FFPeriEvent5_all), (len(np.hstack(FFPeriEvent5_all)),1))
# PeriEventFF_stacked6 = np.reshape(np.hstack(FFPeriEvent6_all), (len(np.hstack(FFPeriEvent6_all)),1))

<<<<<<< Updated upstream
XYZs = np.concatenate((np.vstack(xvals_all_filtered), np.vstack(yvals_all_filtered), np.vstack(zvals_all_filtered)), axis=1)
# PeriEventFR_stackedAll = np.concatenate((PeriEventFR_stacked0, PeriEventFR_stacked1, PeriEventFR_stacked2,
#                                          PeriEventFR_stacked3, PeriEventFR_stacked4, PeriEventFR_stacked5, 
#                                          PeriEventFR_stacked6), axis=1)
PeriEventFF_stackedAll = np.concatenate((PeriEventFF_stacked0, PeriEventFF_stacked1, PeriEventFF_stacked2,
                                         PeriEventFF_stacked3, PeriEventFF_stacked4, PeriEventFF_stacked5, 
                                         PeriEventFF_stacked6), axis=1)
=======
# PeriEventFF_stackedAll = np.concatenate(ß(PeriEventFF_stacked0, PeriEventFF_stacked1, PeriEventFF_stacked2,
#                                          PeriEventFF_stacked3, PeriEventFF_stacked4, PeriEventFF_stacked5, 
#                                          PeriEventFF_stacked6), axis=1)

>>>>>>> Stashed changes
# %% Find the center of mass of target brain regions and substract from xyz

# %% Find the planned repeated site trajectories
rep_traj = one.alyx.rest('trajectories', 'list', provenance='Planned', x=-2243, y=-2000, theta=15)
rep_traj = rep_traj[0]

# Create an insertion object from the trajectory
rep_ins = atlas.Insertion.from_dict(rep_traj, brain_atlas=ba)
# Find the xyz coords of tip of the insertion and the point where the insertion enters the brain
xyz = np.c_[rep_ins.tip, rep_ins.entry].T
# Find the brain regions along the insertion
brain_regions, _ = histology.get_brain_regions(xyz, brain_atlas=ba)

# Find the centre of mass of region
idx_reg = np.where(brain_regions.acronym == regions)[0]
xyz_reg = brain_regions.xyz[idx_reg] #The coordinates we were planning to hit within the given brain region
centre_of_mass_reg = np.mean(xyz_reg, 0)

DeltaX = [((x) - centre_of_mass_reg[0])*1e6 for x in X_stacked]
DeltaY = [(x - centre_of_mass_reg[1])*1e6 for x in Y_stacked]
DeltaZ = [(x - centre_of_mass_reg[2])*1e6 for x in Z_stacked]

# %% 3D Plots

<<<<<<< Updated upstream
cm = plt.get_cmap("viridis")  #viridis, vlag, Accent, cool

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d') 
p = ax.scatter(DeltaX, DeltaY, DeltaZ, c=LabNum_stacked,cmap=cm, depthshade=False, s=4)
#p = ax.scatter(DeltaX, DeltaY, DeltaZ, c=PeriEventFR_stacked0,cmap=cm, depthshade=False, s=4)
=======
# %% Save data and make plots

# # If needed, use codes below to save DataFrame and save as .mat file for Matlab
# ClusterFeatures.to_pickle(FIG_PATH+'/'+regions+'/'+regions +"_ClusterFeatures_DF.pkl")
# # output = pd.read_pickle("ClusterFeatures_DF.pkl")
# # print(output)
# sio.savemat(FIG_PATH+'/'+regions+'/'+regions +'_ClusterFeatMat.mat', {name: col.values for name, col in ClusterFeatures.items()})

regionTitle = regions
cm = plt.get_cmap("viridis")  #viridis, vlag, Accent, cool

fig = plt.figure()
fig.suptitle(regions + ': LabID')
ax = fig.add_subplot(111, projection='3d') 
p = ax.scatter(DeltaX, DeltaY, DeltaZ, c=np.array(ClusterFeatures['LabID'], dtype = np.float64),cmap=cm, depthshade=False, s=4)
>>>>>>> Stashed changes
ax.set_xlabel('dX') #delta x (distance from center of mass of brain region)
ax.set_ylabel('dY')
ax.set_zlabel('dZ')  
cbar = plt.colorbar(p)
if SaveFigs==1:
    plt.savefig(join(FIG_PATH, regions, 'cluster 3D plots Lab ID'))

# #plot center of mass:
# p = ax.scatter([0],[0],[0], c='r', cmap=cm, depthshade=False, s=10)
# ax.plot([0], [0], [0], markerfacecolor='k', markeredgecolor='k', marker='o', markersize=5, alpha=0.6)
plt.show()
<<<<<<< Updated upstream
#plt.hold(True)
#https://stackoverflow.com/questions/21465988/python-equivalent-to-hold-on-in-matlab
=======

fig = plt.figure()
fig.suptitle(regions + ': Perievent FR')
ax = fig.add_subplot(111, projection='3d') 
p = ax.scatter(DeltaX, DeltaY, DeltaZ, c=np.array(PeriEventFRandFF['FR_event0'], dtype = np.float64),cmap=cm, depthshade=False, s=4)
ax.set_xlabel('dX') #delta x (distance from center of mass of brain region)
ax.set_ylabel('dY')
ax.set_zlabel('dZ')  
cbar = plt.colorbar(p)
if SaveFigs==1:
    plt.savefig(join(FIG_PATH, regions, 'cluster 3D plots peri-event FR'))

>>>>>>> Stashed changes

fig = plt.figure()
fig.suptitle(regions + ': Perievent FF')
ax = fig.add_subplot(111, projection='3d') 
p = ax.scatter(DeltaX, DeltaY, DeltaZ, c=np.array(PeriEventFRandFF['FF_event0'], dtype = np.float64),cmap=cm, depthshade=False, s=4)
ax.set_xlabel('dX') #delta x (distance from center of mass of brain region)
ax.set_ylabel('dY')
ax.set_zlabel('dZ')  
cbar = plt.colorbar(p)
if SaveFigs==1:
    plt.savefig(join(FIG_PATH, regions, 'cluster 3D plots peri-event FF'))
    

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d') 
<<<<<<< Updated upstream
p = ax.scatter(DeltaX, DeltaY, DeltaZ, c=amps_stacked,cmap=cm, depthshade=False, s=4)
#p = ax.scatter(DeltaX, DeltaY, DeltaZ, c=PeriEventFR_stacked0,cmap=cm, depthshade=False, s=4)
ax.set_xlabel('dX') #delta x (distance from center of mass of brain region)
ax.set_ylabel('dY')
ax.set_zlabel('dZ')  
cbar = plt.colorbar(p)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d') 
p = ax.scatter(DeltaX, DeltaY, DeltaZ, c=ptt_stacked,cmap=cm, depthshade=False, s=4)
#p = ax.scatter(DeltaX, DeltaY, DeltaZ, c=PeriEventFR_stacked0,cmap=cm, depthshade=False, s=4)
ax.set_xlabel('dX') #delta x (distance from center of mass of brain region)
ax.set_ylabel('dY')
ax.set_zlabel('dZ')  
cbar = plt.colorbar(p)
=======
p = ax.scatter(DeltaX, DeltaY, DeltaZ, c=np.log(np.array(ClusterFeatures['AvgFR'], dtype = np.float64)),cmap=cm, depthshade=False, s=4)
ax.set_xlabel('dX') 
ax.set_ylabel('dY')
ax.set_zlabel('dZ')  
cbar = plt.colorbar(p)
if SaveFigs==1:
    plt.savefig(join(FIG_PATH, regions, 'cluster 3D plots log avg FR'))


fig = plt.figure()
fig.suptitle(regions + ': WF amp')
ax = fig.add_subplot(111, projection='3d') 
p = ax.scatter(DeltaX, DeltaY, DeltaZ, c=np.array(ClusterFeatures['amps'], dtype = np.float64), cmap=cm, depthshade=False, s=4)
ax.set_xlabel('dX') 
ax.set_ylabel('dY')
ax.set_zlabel('dZ')  
cbar = plt.colorbar(p)
if SaveFigs==1:
    plt.savefig(join(FIG_PATH, regions, 'cluster 3D plots WF amp'))


fig = plt.figure()
fig.suptitle(regions + ': WF peak_trough')
ax = fig.add_subplot(111, projection='3d') 
p = ax.scatter(DeltaX, DeltaY, DeltaZ, c=np.array(ClusterFeatures['peak_trough'], dtype = np.float64),cmap=cm, depthshade=False, s=4)
ax.set_xlabel('dX') 
ax.set_ylabel('dY')
ax.set_zlabel('dZ')  
cbar = plt.colorbar(p)
if SaveFigs==1:
    plt.savefig(join(FIG_PATH, regions, 'cluster 3D plots WF peak-trough'))


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
    plt.savefig(join(FIG_PATH, regions, 'Histogram of cluster positions'))
    print('Figures saved')

# %% Time cours of firing rate and fano factor

FRperTime = activities[0]['tscale']
Clust = 20
FRperi=activities[0]['means'][Clust]
plt.plot(FRperTime, MeanFRoverT)


#These 2 should be the same, check:
MeanFRoverT = activities[0].means[activities[0].cscale == Clust] #mean for 1 cluster over trials for each binned time
FRperi=activities[0]['means'][Clust]


VarFRoverT = square(activities[0].stds[activities[0].cscale == cluster])

#MeanFRoverT = activities[0].means[activities[0].cscale == cluster] #mean for 1 cluster over trials for each binned time
#FFev0.append(var(MeanFRoverT)/mean(MeanFRoverT))



#plt.plot(BinnedSpikes[0])

FFoverT = VarFRoverT/MeanFRoverT

plt.plot(FRperTime, np.squeeze(MeanFRoverT))
plt.plot(FRperTime, np.squeeze(VarFRoverT))
plt.plot(FRperTime, np.squeeze(FFoverT))


# peri_event_time_histogram(
#         spike_times, spike_clusters, events, cluster_id,  # Everything you need for a basic plot
#         t_before=0.2, t_after=0.5, bin_size=0.025, smoothing=0.025, as_rate=True,
#         include_raster=False, n_rasters=None, error_bars='std', ax=None,
#         pethline_kwargs={'color': 'blue', 'lw': 2},
#         errbar_kwargs={'color': 'blue', 'alpha': 0.5},
#         eventline_kwargs={'color': 'black', 'alpha': 0.5},
#         raster_kwargs={'color': 'black', 'lw': 0.5}, **kwargs):

>>>>>>> Stashed changes

# #Code to plot rasters:
# n_rasters=4
# ax.axhline(0., color='black')
# tickheight = plot_edge / len(events[:n_rasters])  # How much space per trace
# tickedges = np.arange(0., -plot_edge - 1e-5, -tickheight)
# clu_spks = spike_times[spike_clusters == cluster_id]
# for i, t in enumerate(events[:n_rasters]):
#     idx = np.bitwise_and(clu_spks >= t - t_before, clu_spks <= t + t_after)
#     event_spks = clu_spks[idx]
#     ax.vlines(event_spks - t, tickedges[i + 1], tickedges[i], **raster_kwargs)
# ax.set_ylabel('Firing Rate' if as_rate else 'Number of spikes', y=0.75)

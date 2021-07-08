
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
  
from ibllib.atlas import atlas, AllenAtlas
from ibllib.pipes import histology

one = ONE()
ba = AllenAtlas()

# Query repeated site trajectories
#traj = query()
#traj = query(behavior=True)
ListOfEIDs = eid_list()

# Initialize dataframe
ClusterFeatures = pd.DataFrame() #['eID', 'Lab ID','clusterID', 'AvgFR', 'AvgFF', 'amps', 'peak-trough', 'Xloc', 'Yloc', 'Zloc']

PeriEventMeanFR_all0, FFPeriEvent0 = [], []
PeriEventMeanFR_all1, FFPeriEvent1 = [], []
PeriEventMeanFR_all2, FFPeriEvent2 = [], []
PeriEventMeanFR_all3, FFPeriEvent3 = [], []
PeriEventMeanFR_all4, FFPeriEvent4 = [], []
PeriEventMeanFR_all5, FFPeriEvent5 = [], []
PeriEventMeanFR_all6, FFPeriEvent6 = [], []

#REGIONS = ['PPC', 'CA1', 'DG', 'LP', 'PO']
regions = 'LP' #[PPC', 'CA1', 'DG', 'LP', 'PO']


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
traj=get_traj(ListOfEIDs)

for i in range(len(traj)):
    print('Processing repeated site recording %d of %d' % (i+1, len(traj)))
    fr, FF, LabNumArray = [], [], []
    xvals, yvals, zvals =[], [], []

    PeriEventMeanFRperClust0, FFPeriEvent0_all = [], []
    PeriEventMeanFRperClust1, FFPeriEvent1_all = [], []
    PeriEventMeanFRperClust2, FFPeriEvent2_all = [], []
    PeriEventMeanFRperClust3, FFPeriEvent3_all = [], []
    PeriEventMeanFRperClust4, FFPeriEvent4_all = [], []
    PeriEventMeanFRperClust5, FFPeriEvent5_all = [], []
    PeriEventMeanFRperClust6, FFPeriEvent6_all = [], []
    #PeriEventMeanFRperClust = [0 for _ in eventTypes] #{x: [] for x in eventTypes}
    
    
    # Load in data
    eid = traj[i]['session']['id']
    try:
        spikes, clusters, channels = bbone.load_spike_sorting_with_channel(eid, one=one)
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
    
    #Find all the relevant brain regions by first combining the regions (e.g. cortical layers):
    BrainRegionsInProbe = combine_regions(clusters[probe]['acronym'])
                                          
  
    # Get relevant cluster ids
    clusterIDs  = clusters[probe]['metrics']['cluster_id'][BrainRegionsInProbe == regions][clusters[probe]['metrics']['label'] == 1]
   
        
    #Find cluster waveform (spike amp & width):
    amps1 = one.load_dataset(eid, 'clusters.amps.npy', collection='alf/{}'.format(probe))
    ptt1 = one.load_dataset(eid, 'clusters.peakToTrough.npy', collection='alf/{}'.format(probe))


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
    #event == 'Move'
    times1stMove = one.load(eid, dataset_types=['trials.firstMovement_times'])[0]
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
        pre_time, post_time = 0.2, 0.4
        if count<5:
            pre_time, post_time = 0.2, 0.4
        elif count<7:
            pre_time, post_time = 0.4, 0.2
        a, b = calculate_peths(spikes[probe]['times'], spikes[probe]['clusters'], np.array(clusterIDs),
                               etime, pre_time=pre_time, post_time=post_time, smoothing=0, bin_size=0.01)
        activities.append(a)
        BinnedSpikes.append(b)
        count += 1
        
    # activities_NotNorm = [a.means for a in activities]
    # activities_Normalized = normalise_act(activities,  spikes, probe, clusterIDs, base_line_times)
    #activities = activities_Normalized
    #After the part above, activities[0].means will be equivalent to activities_Normalized[0] or activities_NotNorm[0]
    
    for k, cluster in enumerate(clusterIDs):
        fr.append(spikes[probe]['times'][spikes[probe]['clusters'] == cluster].shape[0]/ spikes[probe]['times'][-1])
        
        #This part calculates the f.r. in bins of 50 ms, then calculates the variance and Fano Factor:
        SpikesOfCluster = spikes[probe]['times'][spikes[probe]['clusters'] == cluster]
        time_bins = arange(0, spikes[probe]['times'][-1], 0.05)
        SpkIncrements50ms, _ = histogram(SpikesOfCluster, time_bins) #vector containing the # of spikes/counts in each 50 ms increment.

        FF50ms = SpkIncrements50ms.var()/SpkIncrements50ms.mean()
        FF.append(FF50ms)        
           
        LabNumArray.append(LabNum)
    
        xvals.append(clusters[probe]['x'][clusters[probe]['metrics']['cluster_id'] == cluster])
        yvals.append(clusters[probe]['y'][clusters[probe]['metrics']['cluster_id'] == cluster])
        zvals.append(clusters[probe]['z'][clusters[probe]['metrics']['cluster_id'] == cluster])
        
        #Now find the average firing rate around the event only, for each cluster:
        PeriEventMeanFRperClust0.append(mean(activities[0].means[activities[0].cscale == cluster]))

        #Now find the Fano Factor around the event only, for each cluster:
        FFPeriEvent0.append(var(activities[0].means[activities[0].cscale == cluster])/mean(activities[0].means[activities[0].cscale == cluster]))
        FFPeriEvent1.append(var(activities[1].means[activities[1].cscale == cluster])/mean(activities[1].means[activities[1].cscale == cluster]))
        FFPeriEvent2.append(var(activities[2].means[activities[2].cscale == cluster])/mean(activities[2].means[activities[2].cscale == cluster]))
        FFPeriEvent3.append(var(activities[3].means[activities[3].cscale == cluster])/mean(activities[3].means[activities[3].cscale == cluster]))
        FFPeriEvent4.append(var(activities[4].means[activities[4].cscale == cluster])/mean(activities[4].means[activities[4].cscale == cluster]))
        FFPeriEvent5.append(var(activities[5].means[activities[5].cscale == cluster])/mean(activities[5].means[activities[5].cscale == cluster]))
        FFPeriEvent6.append(var(activities[6].means[activities[6].cscale == cluster])/mean(activities[6].means[activities[6].cscale == cluster]))
        
    
    #Save all the data:
    data1 = np.array([np.repeat(eid, len(clusterIDs)), LabNumArray, np.array(clusterIDs), np.array(fr), np.array(FF),
                      amps1[clusterIDs], ptt1[clusterIDs], np.squeeze(xvals), np.squeeze(yvals), np.squeeze(zvals)])
    data1 = np.transpose(data1); #data1.reshape(len(clusterIDs),4)
    df = pd.DataFrame(data1, columns=['eID', 'Lab ID','clusterID', 'AvgFR', 'AvgFF', 'amps', 'peak-trough', 'Xloc', 'Yloc', 'Zloc'],
                      index=np.arange(len(ClusterFeatures), len(ClusterFeatures)+len(clusterIDs))) #index=np.arange(0,len(clusterIDs)))
    
    ClusterFeatures = ClusterFeatures.append(df) 
    
    
    PeriEventMeanFR_all0.append(np.array(PeriEventMeanFRperClust0))
    FFPeriEvent0_all.append(np.array(FFPeriEvent0))
    FFPeriEvent1_all.append(np.array(FFPeriEvent1))
    FFPeriEvent2_all.append(np.array(FFPeriEvent2))
    FFPeriEvent3_all.append(np.array(FFPeriEvent3))
    FFPeriEvent4_all.append(np.array(FFPeriEvent4))
    FFPeriEvent5_all.append(np.array(FFPeriEvent5))
    FFPeriEvent6_all.append(np.array(FFPeriEvent6))


PeriEventFR_stacked0 = np.reshape(np.hstack(PeriEventMeanFR_all0), (len(np.hstack(PeriEventMeanFR_all0)),1))
PeriEventFF_stacked0 = np.reshape(np.hstack(FFPeriEvent0_all), (len(np.hstack(FFPeriEvent0_all)),1))
PeriEventFF_stacked1 = np.reshape(np.hstack(FFPeriEvent1_all), (len(np.hstack(FFPeriEvent1_all)),1))
PeriEventFF_stacked2 = np.reshape(np.hstack(FFPeriEvent2_all), (len(np.hstack(FFPeriEvent2_all)),1))
PeriEventFF_stacked3 = np.reshape(np.hstack(FFPeriEvent3_all), (len(np.hstack(FFPeriEvent3_all)),1))
PeriEventFF_stacked4 = np.reshape(np.hstack(FFPeriEvent4_all), (len(np.hstack(FFPeriEvent4_all)),1))
PeriEventFF_stacked5 = np.reshape(np.hstack(FFPeriEvent5_all), (len(np.hstack(FFPeriEvent5_all)),1))
PeriEventFF_stacked6 = np.reshape(np.hstack(FFPeriEvent6_all), (len(np.hstack(FFPeriEvent6_all)),1))

PeriEventFF_stackedAll = np.concatenate((PeriEventFF_stacked0, PeriEventFF_stacked1, PeriEventFF_stacked2,
                                         PeriEventFF_stacked3, PeriEventFF_stacked4, PeriEventFF_stacked5, 
                                         PeriEventFF_stacked6), axis=1)
# %% Find the center of mass of target brain regions and substract from xyz (modified from Mayo's code)

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


# %% Save DataFrame and 3D Plots
ClusterFeatures.to_pickle("ClusterFeatures_DF.pkl")
# output = pd.read_pickle("ClusterFeatures_DF.pkl")
# print(output)


regionTitle = regions

cm = plt.get_cmap("viridis")  #viridis, vlag, Accent, cool

fig = plt.figure()
fig.suptitle(regions + ': Lab ID')
ax = fig.add_subplot(111, projection='3d') 
p = ax.scatter(DeltaX, DeltaY, DeltaZ, c=np.array(ClusterFeatures['Lab ID'], dtype = np.float64),cmap=cm, depthshade=False, s=4)
#p = ax.scatter(DeltaX, DeltaY, DeltaZ, c=PeriEventFR_stacked0,cmap=cm, depthshade=False, s=4)
ax.set_xlabel('dX') #delta x (distance from center of mass of brain region)
ax.set_ylabel('dY')
ax.set_zlabel('dZ')  
cbar = plt.colorbar(p)

# #plot center of mass:
# p = ax.scatter([0],[0],[0], c='r', cmap=cm, depthshade=False, s=10)
# ax.plot([0], [0], [0], markerfacecolor='k', markeredgecolor='k', marker='o', markersize=5, alpha=0.6)
plt.show()

fig = plt.figure()
fig.suptitle(regions + ': Perievent FR')
ax = fig.add_subplot(111, projection='3d') 
p = ax.scatter(DeltaX, DeltaY, DeltaZ, c=PeriEventFR_stacked0,cmap=cm, depthshade=False, s=4)
ax.set_xlabel('dX') #delta x (distance from center of mass of brain region)
ax.set_ylabel('dY')
ax.set_zlabel('dZ')  
cbar = plt.colorbar(p)
plt.savefig(join(FIG_PATH, regions, '3D plots of cluster peri-event FR'))


fig = plt.figure()
fig.suptitle(regions + ': WF amp')
ax = fig.add_subplot(111, projection='3d') 
p = ax.scatter(DeltaX, DeltaY, DeltaZ, c=np.array(ClusterFeatures['amps'], dtype = np.float64), cmap=cm, depthshade=False, s=4)
#p = ax.scatter(DeltaX, DeltaY, DeltaZ, c=PeriEventFR_stacked0,cmap=cm, depthshade=False, s=4)
ax.set_xlabel('dX') #delta x (distance from center of mass of brain region)
ax.set_ylabel('dY')
ax.set_zlabel('dZ')  
cbar = plt.colorbar(p)
plt.savefig(join(FIG_PATH, regions, '3D plots of cluster amp.'))


fig = plt.figure()
fig.suptitle(regions + ': WF peak-trough')
ax = fig.add_subplot(111, projection='3d') 
p = ax.scatter(DeltaX, DeltaY, DeltaZ, c=np.array(ClusterFeatures['peak-trough'], dtype = np.float64),cmap=cm, depthshade=False, s=4)
#p = ax.scatter(DeltaX, DeltaY, DeltaZ, c=ptt_stacked,cmap=cm, depthshade=False, s=4)
ax.set_xlabel('dX') #delta x (distance from center of mass of brain region)
ax.set_ylabel('dY')
ax.set_zlabel('dZ')  
cbar = plt.colorbar(p)
plt.savefig(join(FIG_PATH, regions, '3D plots of cluster peak-trough'))


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
plt.savefig(join(FIG_PATH, regions, 'Histogram of cluster locations'))



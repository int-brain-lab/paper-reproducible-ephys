from one.api import ONE, One
from one.alf.exceptions import ALFObjectNotFound
from reproducible_ephys_functions import combine_regions, BRAIN_REGIONS
from ibllib.atlas import AllenAtlas
from brainbox.io.one import SpikeSortingLoader
from brainbox.behavior import dlc
from brainbox.behavior import wheel as wh
import numpy as np
from figure5.figure5_functions import bin_spikes, bin_spikes2D


one_online = ONE()
one_local = One()
ba = AllenAtlas()


eid = '56b57c38-2699-4091-90a8-aba35103155e'
pid = 'ce397420-3cd2-4a55-8fd1-5e28321981f4'
probe = 'probe01'

# Load in spike sorting data
sl = SpikeSortingLoader(eid=eid, pname=probe, one=one_local, atlas=ba)
spikes, clusters, channels = sl.load_spike_sorting(dataset_types=['clusters.amps', 'clusters.peakToTrough'])
clusters = sl.merge_clusters(spikes, clusters, channels)
clusters['rep_site_acronym'] = combine_regions(clusters['acronym'])
# Find clusters that are in the repeated site brain regions and that have been labelled as good
cluster_idx = np.where(np.bitwise_and(np.isin(clusters['rep_site_acronym'], BRAIN_REGIONS), clusters['label'] == 1))[0]
# cluster_idx = np.where(clusters['label'] == 1)[0]
# Find 20 clusters with highest firing rate
# cluster_idx_fr = np.argsort(clusters['firing_rate'][cluster_idx])[::-1][:20]
# # Find 10 clusters with the highest presence ratio
# cluster_idx_pr = np.argsort(clusters['presence_ratio'][cluster_idx[cluster_idx_fr]])[::-1][0:10]
# # Find the index of the clusters to keep
# cluster_idx = np.sort(cluster_idx[cluster_idx_fr[cluster_idx_pr]]) # need to sort
cluster_id = clusters['cluster_id'][cluster_idx]
# Find the index of spikes that belong to the chosen clusters
spike_idx = np.isin(spikes['clusters'], cluster_id)


# Load in trials data
trials = one_local.load_object(eid, 'trials')
# Find trials that have nans in them
nan_trials = np.c_[np.isnan(trials['stimOn_times']), np.isnan(trials['goCue_times']), np.isnan(trials['response_times']),
                   np.isnan(trials['feedback_times']), np.isnan(trials['stimOff_times']), np.isnan(trials['probabilityLeft'])]
nan_trials = np.sum(nan_trials, axis=1) > 0

# Trials that last longer than 5 seconds
rm_trials = (trials['intervals'][:, 1] - trials['intervals'][:, 0]) > 5

# # Find trials that are too short or too long
# stim_diff = trials['stimOff_times'] - trials['stimOn_times']
# rm_trials = np.bitwise_or(stim_diff < 0, stim_diff > 5)
# Remove these trials from trials object
rm_trials = np.bitwise_or(rm_trials, nan_trials)
trials = {key: trials[key][~rm_trials] for key in trials.keys()}


# Load in left camera dlc data
left_dlc = one_local.load_object(eid, 'leftCamera', attribute=['dlc', 'features', 'times', 'ROIMotionEnergy'], collection='alf')
left_dlc['dlc'] = dlc.likelihood_threshold(left_dlc['dlc'])
# Compute left camera specific features, nose tip velocity, paw_l velocity, paw_r velocity and pupil diameter
left_dlc['vel_nose'] = dlc.get_speed(left_dlc['dlc'], left_dlc['times'], camera='left', feature='nose_tip')
left_dlc['vel_paw_l'] = dlc.get_speed(left_dlc['dlc'], left_dlc['times'], camera='left', feature='paw_l')
left_dlc['vel_paw_r'] = dlc.get_speed(left_dlc['dlc'], left_dlc['times'], camera='left', feature='paw_r')
# if 'features' in left_dlc.keys():
#     left_dlc['pupil_diameter'] = left_dlc.pop('features')['pupilDiameter_smooth']
# else:
#     left_dlc['pupil_diameter'] = dlc.get_smooth_pupil_diameter(dlc.get_pupil_diameter(left_dlc['dlc']), 'left')

# Load in right camera dlc data
right_dlc = one_local.load_object(eid, 'rightCamera', attribute=['dlc', 'features', 'times', 'ROIMotionEnergy'], collection='alf')
right_dlc['dlc'] = dlc.likelihood_threshold(right_dlc['dlc'])
# right_dlc['times'] = adjust_times(eid, right_dlc) # TODO
# Compute right camera specific features, nose tip velocity, paw_l velocity, paw_r velocity and pupil diameter
right_dlc['vel_nose'] = dlc.get_speed(right_dlc['dlc'], right_dlc['times'], camera='right', feature='nose_tip')
right_dlc['vel_paw_l'] = dlc.get_speed(right_dlc['dlc'], right_dlc['times'], camera='right', feature='paw_l')
right_dlc['vel_paw_r'] = dlc.get_speed(right_dlc['dlc'], right_dlc['times'], camera='right', feature='paw_r')

# if 'features' in right_dlc.keys():
#     right_dlc['pupil_diameter'] = right_dlc.pop('features')['pupilDiameter_smooth']
# else:
#     right_dlc['pupil_diameter'] = dlc.get_smooth_pupil_diameter(dlc.get_pupil_diameter(right_dlc['dlc']), 'right')

# Load in lick times (compute if not available)
try:
    licks = one_local.load_object(eid, 'licks', collection='alf')
except ALFObjectNotFound:
    licks_left = dlc.get_licks(left_dlc['dlc'], left_dlc['times'])
    licks_right = dlc.get_licks(right_dlc['dlc'], right_dlc['times'])
    licks = np.sort(np.r_[licks_left, licks_right])

# Load in wheel data
wheel = one_local.load_object(eid, 'wheel')
# Compute wheel velocity and remove nan values
wheel['vel'] = wh.velocity(wheel['timestamps'], wheel['position'])
wheel['vel'] = wheel['vel'][~np.isnan(wheel['vel'])]
wheel['timestamps'] = wheel['timestamps'][~np.isnan(wheel['vel'])]

passive_stim = one_local.load_dataset(eid, '_ibl_passiveGabor.table.csv')
passive_intervals = np.c_[passive_stim['start'].values[:-1], passive_stim['start'].values[1:]]
passive_events = one_local.load_dataset(eid, '_ibl_passiveStims.table.csv')
print('passive data loaded')

all_trials_intervals = np.r_[trials['intervals'], passive_intervals]

# Now prepare our output and features arrays
bin_size = 0.05
pre_time = 0
post_time = 5
n_units = cluster_idx.size
n_active_trials = trials['intervals'].shape[0]
n_passive_trials = passive_intervals.shape[0]
n_trials = n_active_trials + n_passive_trials
n_bins = np.int32((pre_time + post_time) / bin_size)
n_labs = 6
n_regions = len(BRAIN_REGIONS)
event_time = all_trials_intervals[:, 0]


# add the passive and non passive trials together

outputs, t = bin_spikes2D(spikes['times'][spike_idx], spikes['clusters'][spike_idx], cluster_id, event_time,
                          pre_time, post_time, bin_size)
# set values in all bins that are outside the trial interval to 0 to make sure we don't have info from the next trial
t_cutoff = np.searchsorted(t, all_trials_intervals[:, 1] - all_trials_intervals[:, 0])
outputs = truncate(outputs, t_cutoff)
outputs = np.swapaxes(outputs, 0, 1) / bin_size

# outputs_passive, t_pas = bin_spikes2D(spikes['times'][spike_idx], spikes['clusters'][spike_idx], cluster_id, passive_intervals[:, 0],
#                                       pre_time, post_time, bin_size)
# t_cutoff_passive = np.searchsorted(t_pas, passive_intervals[:, 1] - passive_intervals[:, 0])
# outputs_passive = truncate(outputs_passive, t_cutoff_passive)
# outputs_passive = np.swapaxes(outputs_passive, 0, 1) / bin_size


# Now compute the features - these are the values that we use to predict the outputs
features = {}

# Lab
lab_idx = 2
features['lab'] = np.zeros((n_units, n_trials, n_bins, n_labs))
features['lab'][:, :, :, lab_idx] = 1

# Session
# features['']

# Cluster acronym
acronym_idx = 0
# features['acronym'] = np.ones(n_units, n_trials, n_bins, n_regions) * acronym_idx

# Cluster 3D spatial position
features['spatial_position_x'] = np.swapaxes(np.repeat(np.array([np.repeat(np.array([clusters['x'][cluster_id]]),
                                                           n_trials, axis=0)]), n_bins, axis=0), 0, 2)[:, :, :, np.newaxis]
features['spatial_position_y'] = np.swapaxes(np.repeat(np.array([np.repeat(np.array([clusters['y'][cluster_id]]),
                                                           n_trials, axis=0)]), n_bins, axis=0), 0, 2)[:, :, :, np.newaxis]
features['spatial_position_z'] = np.swapaxes(np.repeat(np.array([np.repeat(np.array([clusters['z'][cluster_id]]),
                                                           n_trials, axis=0)]), n_bins, axis=0), 0, 2)[:, :, :, np.newaxis]

# Cluster amplitude
features['unit_amplitude'] = np.swapaxes(np.repeat(np.array([np.repeat(np.array([clusters['amps'][cluster_id]]),
                                                           n_trials, axis=0)]), n_bins, axis=0), 0, 2)[:, :, :, np.newaxis]

# Cluster peaktotrough
features['unit_waveform_width'] = np.swapaxes(np.repeat(np.array([np.repeat(np.array([clusters['peakToTrough'][cluster_id]]),
                                                           n_trials, axis=0)]), n_bins, axis=0), 0, 2)[:, :, :, np.newaxis]

# Wheel velocity
binned_val, _ = bin_spikes(wheel['timestamps'], event_time, pre_time, post_time, bin_size, weights=wheel['vel'])
# To weight by mean value in each bin rather than the sum of values in each bin
binned_count, _ = bin_spikes(wheel['timestamps'], event_time, pre_time, post_time, bin_size)
binned_count[binned_count == 0] = 1
binned_val = binned_val / binned_count
binned_val = truncate(binned_val, t_cutoff)
features['wheel_velocity'] = np.repeat(binned_val[np.newaxis], n_units, axis=0)[:, :, :,  np.newaxis]

# Left dlc
# Nose tip velocity
binned_val, _ = bin_spikes(left_dlc['times'], event_time, pre_time, post_time, bin_size,
                           weights=left_dlc['vel_nose'])
binned_val = truncate(binned_val, t_cutoff)
features['left_nose_speed'] = np.repeat(binned_val[np.newaxis], n_units, axis=0)[:, :, :,  np.newaxis]
# Paw left velocity
binned_val, _ = bin_spikes(left_dlc['times'], event_time, pre_time, post_time, bin_size,
                           weights=left_dlc['vel_paw_l'])
binned_val = truncate(binned_val, t_cutoff)
features['left_pawL_speed'] = np.repeat(binned_val[np.newaxis], n_units, axis=0)[:, :, :,  np.newaxis]
# Paw right velocity
binned_val, _ = bin_spikes(left_dlc['times'], event_time, pre_time, post_time, bin_size,
                           weights=left_dlc['vel_paw_r'])
binned_val = truncate(binned_val, t_cutoff)
features['left_pawR_speed'] = np.repeat(binned_val[np.newaxis], n_units, axis=0)[:, :, :,  np.newaxis]
# Pupil diameter
# binned_val, _ = bin_spikes(left_dlc['times'], trials['stimOn_times'], pre_time, post_time, bin_size,
#                            weights=left_dlc['pupil_diameter'])
# features['left_pupil_diameter'] = np.swapaxes(np.repeat(binned_val[:, np.newaxis], n_units, axis=1), 0, 1)[:, :, :,  np.newaxis]
# ROI Motion energy
binned_val, _ = bin_spikes(left_dlc['times'], event_time, pre_time, post_time, bin_size,
                           weights=left_dlc['ROIMotionEnergy'])
binned_val = truncate(binned_val, t_cutoff)
features['left_motion_energy'] = np.repeat(binned_val[np.newaxis], n_units, axis=0)[:, :, :,  np.newaxis]


# Right dlc
# Nose tip velocity
binned_val, _ = bin_spikes(right_dlc['times'], event_time, pre_time, post_time, bin_size,
                           weights=right_dlc['vel_nose'])
binned_val = truncate(binned_val, t_cutoff)
features['right_nose_speed'] = np.repeat(binned_val[np.newaxis], n_units, axis=0)[:, :, :,  np.newaxis]
# Paw left velocity
binned_val, _ = bin_spikes(right_dlc['times'], event_time, pre_time, post_time, bin_size,
                           weights=right_dlc['vel_paw_l'])
binned_val = truncate(binned_val, t_cutoff)
features['right_pawL_speed'] = np.repeat(binned_val[np.newaxis], n_units, axis=0)[:, :, :,  np.newaxis]
# Paw right velocity
binned_val, _ = bin_spikes(right_dlc['times'], event_time, pre_time, post_time, bin_size,
                           weights=right_dlc['vel_paw_r'])
binned_val = truncate(binned_val, t_cutoff)
features['right_pawR_speed'] = np.repeat(binned_val[np.newaxis], n_units, axis=0)[:, :, :,  np.newaxis]
# Pupil diameter
# binned_val, _ = bin_spikes(right_dlc['times'], event_time, pre_time, post_time, bin_size,
#                            weights=right_dlc['pupil_diameter'])
# features['right_pupil_diameter'] = np.swapaxes(np.repeat(binned_val[:, np.newaxis], n_units, axis=1), 0, 1)[:, :, :,  np.newaxis]
# # ROI Motion energy
# binned_val, _ = bin_spikes(right_dlc['times'], event_time, pre_time, post_time, bin_size,
#                            weights=right_dlc['ROIMotionEnergy'])
# features['right_motion_energy'] = np.swapaxes(np.repeat(binned_val[:, np.newaxis], n_units, axis=1), 0, 1)[:, :, :,  np.newaxis]

# Licks
binned_val, _ = bin_spikes(licks, event_time, pre_time, post_time, bin_size)
binned_val = truncate(binned_val, t_cutoff)
features['lick'] = np.repeat(binned_val[np.newaxis], n_units, axis=0)[:, :, :,  np.newaxis]

# Choice
binned_val, _ = bin_spikes(trials['response_times'], event_time, pre_time, post_time, bin_size)
binned_val = truncate(binned_val, t_cutoff)
left_choice = np.where(trials['choice'] == -1)[0]
no_choice = np.where(trials['choice'] == 0)[0]
right_choice = np.where(trials['choice'] == 1)[0]
features['choice'] = np.zeros((n_units, n_trials, n_bins, 3))
features['choice'][:, left_choice, :, 0] = np.repeat(binned_val[left_choice, :][:, np.newaxis], n_units, axis=1)
features['choice'][:, no_choice, :, 1] = np.repeat(binned_val[no_choice, :][:, np.newaxis], n_units, axis=1)
features['choice'][:, right_choice, :, 2] = np.repeat(binned_val[right_choice, :][:, np.newaxis], n_units, axis=1)

# Feedback
binned_val, _ = bin_spikes(trials['feedback_times'], event_time, pre_time, post_time, bin_size)
binned_val = truncate(binned_val, t_cutoff)
incorrect = np.where(trials['feedbackType'] == -1)[0]
correct = np.where(trials['feedbackType'] == 1)[0]
features['feedback'] = np.zeros((n_units, n_trials, n_bins, 2))
features['feedback'][:, incorrect, :, 0] = np.repeat(binned_val[incorrect, :][:, np.newaxis], n_units, axis=1)
features['feedback'][:, correct, :, 1] = np.repeat(binned_val[correct, :][:, np.newaxis], n_units, axis=1)

# Go Cue
binned_val, _ = bin_spikes(trials['goCue_times'], event_time, pre_time, post_time, bin_size)
binned_val = truncate(binned_val, t_cutoff)
features['go_cue'] = np.swapaxes(np.repeat(binned_val[:, np.newaxis], n_units, axis=1), 0, 1)[:, :, :,  np.newaxis]

# Stim On
# Find location of stim on bin
binned_val1, _ = bin_spikes(trials['stimOn_times'], event_time, pre_time, post_time, bin_size)
binned_val1 = truncate(binned_val1, t_cutoff)
val1 = np.where(binned_val1 == 1)

binned_val2, _ = bin_spikes(trials['stimOff_times'], event_time, pre_time, post_time, bin_size)
binned_val2 = truncate(binned_val2, t_cutoff)
val2 = np.where(binned_val2 == 1)

assert np.array_equal(val1[0], val2[0])

binned_val = np.zeros((n_trials, n_bins))
for idx, v1, v2 in zip(val1[0], val1[1], val2[1]):
    binned_val[idx, v1:(v2+1)] = 1

features['stimulus_on'] = np.repeat(binned_val[np.newaxis], n_units, axis=0)[:, :, :,  np.newaxis]

# Contrast
left_contrast = np.where(~np.isnan(trials['contrastLeft']))[0]
right_contrast = np.where(~np.isnan(trials['contrastRight']))[0]
features['contrast'] = np.zeros((n_units, n_trials, n_bins, 2))
features['contrast'][:, left_contrast, :, 0] = np.repeat((binned_val[left_contrast, :] *
                                                          trials['contrastLeft'][left_contrast][:, np.newaxis])[:, np.newaxis],
                                                         n_units, axis=1)
features['contrast'][:, right_contrast, :, 1] = np.repeat((binned_val[right_contrast, :] *
                                                           trials['contrastRight'][right_contrast][:, np.newaxis])[:, np.newaxis],
                                                          n_units, axis=1)

# Probability Left
features['prob_left'] = np.swapaxes(np.repeat((binned_val * trials['probabilityLeft'][:, np.newaxis])[:,np.newaxis],
                                              n_units, axis=1), 0, 1)[:, :, :,  np.newaxis]


# Now for passive trials
# passive valve on / off
binned_val1, _ = bin_spikes(passive_events['valveOn'], event_time, pre_time, post_time, bin_size)
binned_val1 = truncate(binned_val1, t_cutoff)
val1 = np.where(binned_val1 == 1)

binned_val2, _ = bin_spikes(passive_events['valveOff'], event_time, pre_time, post_time, bin_size)
binned_val2 = truncate(binned_val2, t_cutoff)
val2 = np.where(binned_val2 == 1)

assert np.array_equal(val1[0], val2[0])

binned_val = np.zeros((n_trials, n_bins))
for idx, v1, v2 in zip(val1[0], val1[1], val2[1]):
    binned_val[idx, v1:(v2+1)] = 1

# passive tone on / off
binned_val1, _ = bin_spikes(passive_events['toneOn'], event_time, pre_time, post_time, bin_size)
binned_val1 = truncate(binned_val1, t_cutoff)
val1 = np.where(binned_val1 == 1)

binned_val2, _ = bin_spikes(passive_events['toneOff'], event_time, pre_time, post_time, bin_size)
binned_val2 = truncate(binned_val2, t_cutoff)
val2 = np.where(binned_val2 == 1)

assert np.array_equal(val1[0], val2[0])

binned_val = np.zeros((n_trials, n_bins))
for idx, v1, v2 in zip(val1[0], val1[1], val2[1]):
    binned_val[idx, v1:(v2+1)] = 1

# passive noise on/ off
binned_val1, _ = bin_spikes(passive_events['noiseOn'], event_time, pre_time, post_time, bin_size)
binned_val1 = truncate(binned_val1, t_cutoff)
val1 = np.where(binned_val1 == 1)

binned_val2, _ = bin_spikes(passive_events['noiseOff'], event_time, pre_time, post_time, bin_size)
binned_val2 = truncate(binned_val2, t_cutoff)
val2 = np.where(binned_val2 == 1)

assert np.array_equal(val1[0], val2[0])

binned_val = np.zeros((n_trials, n_bins))
for idx, v1, v2 in zip(val1[0], val1[1], val2[1]):
    binned_val[idx, v1:(v2+1)] = 1


# Now we build up our mega array





def truncate(vals, cutoff_vals):
    if vals.ndim == 3:
        for i in range(cutoff_vals.size):
            vals[i, :, cutoff_vals[i]:] = 0
    elif vals.ndim == 2:
        for i in range(cutoff_vals.size):
            vals[i, cutoff_vals[i]:] = 0

    return vals





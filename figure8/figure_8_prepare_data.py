import numpy as np

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
# cluster_idx = np.where(np.bitwise_and(np.isin(clusters['rep_site_acronym'], BRAIN_REGIONS), clusters['label'] == 1))[0]
cluster_idx = np.where(clusters['label'] == 1)[0]
# Find 20 clusters with highest firing rate
cluster_idx_fr = np.argsort(clusters['firing_rate'][cluster_idx])[::-1][:20]
# Find 10 clusters with the highest presence ratio
cluster_idx_pr = np.argsort(clusters['presence_ratio'][cluster_idx[cluster_idx_fr]])[::-1][0:10]
# Find the index of the clusters to keep
cluster_idx = np.sort(cluster_idx[cluster_idx_fr[cluster_idx_pr]]) # need to sort
cluster_id = clusters['cluster_id'][cluster_idx]
# Find the index of spikes that belong to the chosen clusters
spike_idx = np.isin(spikes['clusters'], cluster_id)


# Load in trials data
trials = one_local.load_object(eid, 'trials')
# Find trials that have nans in them
nan_trials = np.c_[np.isnan(trials['stimOn_times']), np.isnan(trials['goCue_times']), np.isnan(trials['response_times']),
                   np.isnan(trials['feedback_times']), np.isnan(trials['stimOff_times']), np.isnan(trials['probabilityLeft'])]
nan_trials = np.sum(nan_trials, axis=1) > 0
# Find trials that are too short or too long
stim_diff = trials['stimOff_times'] - trials['stimOn_times']
rm_trials = np.bitwise_or(stim_diff < 0, stim_diff > 5)
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


# Now prepare our output and features arrays
bin_size = 0.05
pre_time = 0.2
post_time = 2

# Now compute the outputs - these are the actual values that we want our model to predict
outputs, _ = bin_spikes2D(spikes['times'][spike_idx], spikes['clusters'][spike_idx], cluster_id, trials['stimOn_times'],
                       pre_time, post_time, bin_size)
outputs = np.swapaxes(outputs, 0, 1) / bin_size  # convert to firing rate

# Now compute the features - these are the values that we use to predict the outputs
n_units = cluster_idx.size
n_trials = trials['stimOn_times'].size
n_bins = np.int32((pre_time + post_time) / bin_size)
n_labs = 6
n_regions = len(BRAIN_REGIONS)

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
binned_val, _ = bin_spikes(wheel['timestamps'], trials['stimOn_times'], pre_time, post_time, bin_size, weights=wheel['vel'])
# To weight by mean value in each bin rather than the sum of values in each bin
binned_count, _ = bin_spikes(wheel['timestamps'], trials['stimOn_times'], pre_time, post_time, bin_size)
binned_count[binned_count == 0] = 1
binned_val = binned_val / binned_count
features['wheel_velocity'] = np.swapaxes(np.repeat(binned_val[:, np.newaxis], n_units, axis=1), 0, 1)[:, :, :,  np.newaxis]

# Left dlc
# Nose tip velocity
binned_val, _ = bin_spikes(left_dlc['times'], trials['stimOn_times'], pre_time, post_time, bin_size,
                           weights=left_dlc['vel_nose'])
features['left_nose_speed'] = np.swapaxes(np.repeat(binned_val[:, np.newaxis], n_units, axis=1), 0, 1)[:, :, :,  np.newaxis]
# Paw left velocity
binned_val, _ = bin_spikes(left_dlc['times'], trials['stimOn_times'], pre_time, post_time, bin_size,
                           weights=left_dlc['vel_paw_l'])
features['left_pawL_speed'] = np.swapaxes(np.repeat(binned_val[:, np.newaxis], n_units, axis=1), 0, 1)[:, :, :,  np.newaxis]
# Paw right velocity
binned_val, _ = bin_spikes(left_dlc['times'], trials['stimOn_times'], pre_time, post_time, bin_size,
                           weights=left_dlc['vel_paw_r'])
features['left_pawR_speed'] = np.swapaxes(np.repeat(binned_val[:, np.newaxis], n_units, axis=1), 0, 1)[:, :, :,  np.newaxis]
# Pupil diameter
# binned_val, _ = bin_spikes(left_dlc['times'], trials['stimOn_times'], pre_time, post_time, bin_size,
#                            weights=left_dlc['pupil_diameter'])
# features['left_pupil_diameter'] = np.swapaxes(np.repeat(binned_val[:, np.newaxis], n_units, axis=1), 0, 1)[:, :, :,  np.newaxis]
# ROI Motion energy
binned_val, _ = bin_spikes(left_dlc['times'], trials['stimOn_times'], pre_time, post_time, bin_size,
                           weights=left_dlc['ROIMotionEnergy'])
features['left_motion_energy'] = np.swapaxes(np.repeat(binned_val[:, np.newaxis], n_units, axis=1), 0, 1)[:, :, :,  np.newaxis]


# Right dlc
# Nose tip velocity
binned_val, _ = bin_spikes(right_dlc['times'], trials['stimOn_times'], pre_time, post_time, bin_size,
                           weights=right_dlc['vel_nose'])
features['right_nose_speed'] = np.swapaxes(np.repeat(binned_val[:, np.newaxis], n_units, axis=1), 0, 1)[:, :, :,  np.newaxis]
# Paw left velocity
binned_val, _ = bin_spikes(right_dlc['times'], trials['stimOn_times'], pre_time, post_time, bin_size,
                           weights=right_dlc['vel_paw_l'])
features['right_pawL_speed'] = np.swapaxes(np.repeat(binned_val[:, np.newaxis], n_units, axis=1), 0, 1)[:, :, :,  np.newaxis]
# Paw right velocity
binned_val, _ = bin_spikes(right_dlc['times'], trials['stimOn_times'], pre_time, post_time, bin_size,
                           weights=right_dlc['vel_paw_r'])
features['right_pawR_speed'] = np.swapaxes(np.repeat(binned_val[:, np.newaxis], n_units, axis=1), 0, 1)[:, :, :,  np.newaxis]
# Pupil diameter
# binned_val, _ = bin_spikes(right_dlc['times'], trials['stimOn_times'], pre_time, post_time, bin_size,
#                            weights=right_dlc['pupil_diameter'])
# features['right_pupil_diameter'] = np.swapaxes(np.repeat(binned_val[:, np.newaxis], n_units, axis=1), 0, 1)[:, :, :,  np.newaxis]
# # ROI Motion energy
# binned_val, _ = bin_spikes(right_dlc['times'], trials['stimOn_times'], pre_time, post_time, bin_size,
#                            weights=right_dlc['ROIMotionEnergy'])
# features['right_motion_energy'] = np.swapaxes(np.repeat(binned_val[:, np.newaxis], n_units, axis=1), 0, 1)[:, :, :,  np.newaxis]

# Licks
binned_val, _ = bin_spikes(licks, trials['stimOn_times'], pre_time, post_time, bin_size)
features['lick'] = np.swapaxes(np.repeat(binned_val[:, np.newaxis], n_units, axis=1), 0, 1)[:, :, :,  np.newaxis]

# Choice
binned_val, _ = bin_spikes(trials['response_times'], trials['stimOn_times'], pre_time, post_time, bin_size)
left_choice = np.where(trials['choice'] == -1)[0]
no_choice = np.where(trials['choice'] == 0)[0]
right_choice = np.where(trials['choice'] == 1)[0]
features['choice'] = np.zeros((n_units, n_trials, n_bins, 3))
features['choice'][:, left_choice, :, 0] = np.repeat(binned_val[left_choice, :][:, np.newaxis], n_units, axis=1)
features['choice'][:, no_choice, :, 1] = np.repeat(binned_val[no_choice, :][:, np.newaxis], n_units, axis=1)
features['choice'][:, right_choice, :, 2] = np.repeat(binned_val[right_choice, :][:, np.newaxis], n_units, axis=1)

# Feedback
binned_val, _ = bin_spikes(trials['feedback_times'], trials['stimOn_times'], pre_time, post_time, bin_size)
incorrect = np.where(trials['feedbackType'] == -1)[0]
correct = np.where(trials['feedbackType'] == 1)[0]
features['feedback'] = np.zeros((n_units, n_trials, n_bins, 2))
features['feedback'][:, incorrect, :, 0] = np.repeat(binned_val[incorrect, :][:, np.newaxis], n_units, axis=1)
features['feedback'][:, correct, :, 1] = np.repeat(binned_val[correct, :][:, np.newaxis], n_units, axis=1)

# Go Cue
binned_val, _ = bin_spikes(trials['goCue_times'], trials['stimOn_times'], pre_time, post_time, bin_size)
features['go_cue'] = np.swapaxes(np.repeat(binned_val[:, np.newaxis], n_units, axis=1), 0, 1)[:, :, :,  np.newaxis]

# Stim On
# Find location of stim on bin
val1 = np.where(bin_spikes(trials['stimOn_times'], trials['stimOn_times'], pre_time, post_time, bin_size)[0] == 1)[1]
# Find location of stim off bin
val2 = bin_spikes(trials['stimOff_times'], trials['stimOn_times'], pre_time, post_time, bin_size)[0] == 1
# If the stim off bin lies outside the epoch window that we have used, then set the last bin to be the stim off time
val2[np.sum(val2, axis=1) == 0, -1] = True
val2 = np.where(val2)[1]

binned_val = np.zeros((n_trials, n_bins))
for i, (v1, v2) in enumerate(zip(val1, val2)):
    binned_val[i, v1:(v2+1)] = 1

features['stimulus_on'] = np.swapaxes(np.repeat(binned_val[:, np.newaxis], n_units, axis=1), 0, 1)[:, :, :,  np.newaxis]

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

# Now we build up our mega array















"""
functions for getting/processing features for training MTNN
"""

import numpy as np
from oneibl.one import ONE
from reproducible_ephys_functions import query
from tqdm import notebook
import brainbox as bb
import brainbox.io.one as bbone
from collections import defaultdict
import brainbox.behavior.wheel as wh

from ibllib.qc.camera import CameraQC
import os

lab_offset = 1
xyz_offset = 10
max_ptp_offset = 13
wf_width_offset = 14
left_dlc_offset = 15
right_dlc_offset = 29
stimOnOff_offset = 43
contrast_offset = 44
goCue_offset = 46
choice_offset = 47
reward_offset = 50
wheel_offset = 52
pLeft_offset = 53
lick_offset = 54
acronym_offset = 55

def get_new_timestamps(eid, video_type, output_directory):
    '''
    Re-extracting timestamps:
    '''

    if os.path.exists(os.path.join(output_directory,'{}_{}.npy'.format(eid, video_type))):
        print('already exists!')
        return
    
    one = ONE()
    print('one defined')
    qc = CameraQC(eid, video_type, stream=True)
    print('qc defined')
    qc.load_data(extract_times=True)
    print('data loaded')

    ts = qc.data.timestamps 
    np.save(os.path.join(output_directory,'{}_{}.npy'.format(eid, video_type)), ts)
    print('{} new timestamps saved'.format(video_type))

def get_lab_number_map():
    lab_number_map = {'cortexlab': 0, 'mainenlab': 1, 'zadorlab': 2,
                  'churchlandlab': 3, 'angelakilab': 4, 'wittenlab': 5,
                  'hoferlab': 6, 'mrsicflogellab': 6, 'danlab': 7,
                  'steinmetzlab': 8}
    return lab_number_map

# has_GPIO: has the GPIO file, and dlc alignment is resolved
# no_GPIO: does not have the GPIO file; the tuple corresponds to offsets
def get_valid_eids():
    has_GPIO = ['56b57c38-2699-4091-90a8-aba35103155e',
                '7b26ce84-07f9-43d1-957f-bc72aeb730a3',
                'dac3a4c1-b666-4de0-87e8-8c514483cacf',
                '6f09ba7e-e3ce-44b0-932b-c003fb44fb89',
                '54238fd6-d2d0-4408-b1a9-d19d24fd29ce',
                '064a7252-8e10-4ad6-b3fd-7a88a2db5463',
                '4a45c8ba-db6f-4f11-9403-56e06a33dfa4',
                '3638d102-e8b6-4230-8742-e548cd87a949',
                'd0ea3148-948d-4817-94f8-dcaf2342bbbe',
                '7f6b86f9-879a-4ea2-8531-294a221af5d0',
                'd23a44ef-1402-4ed7-97f5-47e9a7a504d9']
    
    no_GPIO = {'746d1902-fa59-4cab-b0aa-013be36060d5':(-3,-1),
               'ee40aece-cffd-4edb-a4b6-155f158c666a':(-3,-7),
               'e535fb62-e245-4a48-b119-88ce62a6fe67':(-5,-6),
               'b03fbc44-3d8e-4a6c-8a50-5ea3498568e0':(-3,-1),
               'db4df448-e449-4a6f-a0e7-288711e7a75a':(-3,-4),
               '41872d7f-75cb-4445-bb1a-132b354c44f0':(-1,-9),
               'c7248e09-8c0d-40f2-9eb4-700a8973d8c8':(-4,-3),
               'aad23144-0e52-4eac-80c5-c4ee2decb198':(-3,-5)}
    
    return has_GPIO, no_GPIO

def get_traj(has_GPIO, no_GPIO):
    traj = query()
    tmp = []
    for t in traj:
        if t['session']['id'] in has_GPIO or t['session']['id'] in no_GPIO.keys():
            tmp.append(t)
    traj = tmp
    
    return traj

# eids: list
def get_acronym_dict(one, traj, has_GPIO, no_GPIO, output_directory):
    acronyms = set()

    for i in range(len(traj)):
        eid = traj[i]['session']['id']
        print('getting acronyms from eid {}'.format(eid))

        try: # try downloading relevant info
            spikes, clusters, channels = bbone.load_spike_sorting_with_channel(eid, one=one, aligned=True)
            print('spikes retrieved')

            print('loading dlc')
            left_dlc = one.load_dataset(eid, '_ibl_leftCamera.dlc.pqt')
            right_dlc = one.load_dataset(eid, '_ibl_rightCamera.dlc.pqt')

            if eid in has_GPIO:
                left_dlc_times = np.load(os.path.join(output_directory,'{}_{}.npy'.format(eid, video_type)))
                right_dlc_times = np.load(os.path.join(output_directory,'{}_{}.npy'.format(eid, video_type)))
            elif eid in no_GPIO.keys():
                left_dlc_times = one.load_dataset(eid, '_ibl_leftCamera.times.npy')
                right_dlc_times = one.load_dataset(eid, '_ibl_rightCamera.times.npy')

                left_offset, right_offset = no_GPIO[i]
                left_dlc_times = left_dlc_times[abs(left_offset):abs(left_offset)+left_dlc.shape[0]]
                right_dlc_times = right_dlc_times[abs(right_offset):abs(right_offset)+right_dlc.shape[0]]

            print((left_dlc_times.shape[0], left_dlc.shape[0]))
            print((right_dlc_times.shape[0], right_dlc.shape[0]))
                
            assert(left_dlc_times.shape[0] == left_dlc.shape[0])
            assert(right_dlc_times.shape[0] == right_dlc.shape[0])
            print('dlc retrieved')

            stimOn_times = one.load_dataset(eid, '_ibl_trials.stimOn_times.npy')

            contrastLeft = one.load_dataset(eid, '_ibl_trials.contrastLeft.npy')
            contrastRight = one.load_dataset(eid, '_ibl_trials.contrastRight.npy')

            goCue_times = one.load_dataset(eid, '_ibl_trials.goCue_times.npy')

            choice = one.load_dataset(eid, '_ibl_trials.choice.npy')
            response_times = one.load_dataset(eid, '_ibl_trials.response_times.npy')


            feedbackType = one.load_dataset(eid, '_ibl_trials.feedbackType.npy')
            feedback_times = one.load_dataset(eid, '_ibl_trials.feedback_times.npy')

            stimOff_times = one.load_dataset(eid, '_ibl_trials.stimOff_times.npy')

            wheel_positions = one.load_dataset(eid, '_ibl_wheel.position.npy')
            wheel_timestamps = one.load_dataset(eid, '_ibl_wheel.timestamps.npy')

            pLeft = one.load_dataset(eid, '_ibl_trials.probabilityLeft.npy')


            assert(stimOff_times.shape[0] == feedback_times.shape[0])
            assert(stimOff_times.shape[0] == contrastLeft.shape[0])
            assert(stimOff_times.shape[0] == contrastRight.shape[0])
            assert(stimOff_times.shape[0] == goCue_times.shape[0])
            assert(stimOff_times.shape[0] == choice.shape[0])
            assert(stimOff_times.shape[0] == response_times.shape[0])
            assert(stimOff_times.shape[0] == feedbackType.shape[0])
            assert(stimOff_times.shape[0] == pLeft.shape[0])
            print('trial info retrieved')

        except: # relevant info not available
            print('error processing eid {}'.format(eid))
            print('-----------------------------------------------')
            continue

        probe = traj[i]['probe_name']
        good_clusters = clusters[probe]['metrics']['cluster_id'][clusters[probe]['metrics']['label'] == 1].to_numpy()
        good_clusters_fr = clusters[probe]['metrics']['firing_rate'].to_numpy()[good_clusters]
        good_clusters_high_fr_idx = good_clusters_fr.argsort()[::-1][:20]
        good_clusters = good_clusters[good_clusters_high_fr_idx]

        good_clusters_pr = clusters[probe]['metrics']['presence_ratio'].to_numpy()[good_clusters]
        good_clusters_high_pr_idx = good_clusters_pr.argsort()[::-1][:10]
        good_clusters = good_clusters[good_clusters_high_pr_idx]

        print("good cluster id: ", good_clusters)
        # cluster specific
        for j, cluster in notebook.tqdm(enumerate(good_clusters)):

            # acronym
            acronym = clusters[probe]['acronym'][cluster]
            acronyms.add(acronym)

        print('-----------------------------------------------')
        
        
    acronym_dict = {}
    for i, ac in enumerate(acronyms):
        acronym_dict[ac] = i

    return acronym_dict


def get_licks(XYs):

    '''
    define a frame as a lick frame if
    x or y for left or right tongue point
    change more than half the sdt of the diff
    '''

    licks = []
    for point in ['tongue_end_l', 'tongue_end_r']:
        for c in XYs[point]:
            thr = np.nanstd(np.diff(c))/4
            licks.append(set(np.where(abs(np.diff(c))>thr)[0]))
    return sorted(list(set.union(*licks)))

def get_dlc_XYs(dlc):
    points = np.unique(['_'.join(x.split('_')[:-1]) for x in dlc.keys()])

    # Set values to nan if likelyhood is too low # for pqt: .to_numpy()
    XYs = {}
    for point in points:
        x = np.ma.masked_where(
            dlc[point + '_likelihood'] < 0.9, dlc[point + '_x'])
        x = x.filled(np.nan)
        y = np.ma.masked_where(
            dlc[point + '_likelihood'] < 0.9, dlc[point + '_y'])
        y = y.filled(np.nan)
        XYs[point] = np.array(
            [x, y])    

    return XYs   

def get_relevant_columns(data):
    relevant_columns = []
    
    columns = data.columns.to_numpy()
    for column in columns:
        if "likelihood" in column:
            continue
        if "tube" in column:
            continue
        if "tongue" in column:
            continue
        data_column = data[column].to_numpy()
        relevant_columns.append(data_column)
    
    return np.asarray(relevant_columns)

# main function
def featurize(i, trajectory, one, lab_number_map, acronym_dict, has_GPIO, no_GPIO, output_directory, bin_size=0.05):
    try: # download relevant info
        eid = trajectory['session']['id']
        print('processing {}'.format(eid))
        lab_id = lab_number_map[trajectory['session']['lab']]
        print(lab_id)
        spikes, clusters, channels = bbone.load_spike_sorting_with_channel(eid, one=one, aligned=True)
        print('spikes retrieved')

        print('loading dlc')
        left_dlc = one.load_dataset(eid, '_ibl_leftCamera.dlc.pqt')
        right_dlc = one.load_dataset(eid, '_ibl_rightCamera.dlc.pqt')
        
        if eid in has_GPIO:
            left_dlc_times = np.load(os.path.join(output_directory,'{}_{}.npy'.format(eid, video_type)))
            right_dlc_times = np.load(os.path.join(output_directory,'{}_{}.npy'.format(eid, video_type)))
        elif eid in no_GPIO.keys():
            left_dlc_times = one.load_dataset(eid, '_ibl_leftCamera.times.npy')
            right_dlc_times = one.load_dataset(eid, '_ibl_rightCamera.times.npy')
            
            left_offset, right_offset = no_GPIO[i]
            left_dlc_times = left_dlc_times[abs(left_offset):abs(left_offset)+left_dlc.shape[0]]
            right_dlc_times = right_dlc_times[abs(right_offset):abs(right_offset)+right_dlc.shape[0]]
        else:
            return 0, 0, 0, 0, 0, False

        assert(left_dlc_times.shape[0] == left_dlc.shape[0])
        assert(right_dlc_times.shape[0] == right_dlc.shape[0])
        print('dlc retrieved')
        
        stimOn_times = one.load_dataset(eid, '_ibl_trials.stimOn_times.npy')

        contrastLeft = one.load_dataset(eid, '_ibl_trials.contrastLeft.npy')
        contrastRight = one.load_dataset(eid, '_ibl_trials.contrastRight.npy')
        
        goCue_times = one.load_dataset(eid, '_ibl_trials.goCue_times.npy')
        
        choice = one.load_dataset(eid, '_ibl_trials.choice.npy')
        response_times = one.load_dataset(eid, '_ibl_trials.response_times.npy')
        
        
        feedbackType = one.load_dataset(eid, '_ibl_trials.feedbackType.npy')
        feedback_times = one.load_dataset(eid, '_ibl_trials.feedback_times.npy')

        stimOff_times = one.load_dataset(eid, '_ibl_trials.stimOff_times.npy')

        wheel_positions = one.load_dataset(eid, '_ibl_wheel.position.npy')
        wheel_timestamps = one.load_dataset(eid, '_ibl_wheel.timestamps.npy')
        
        pLeft = one.load_dataset(eid, '_ibl_trials.probabilityLeft.npy')
        
        
        assert(stimOff_times.shape[0] == feedback_times.shape[0])
        assert(stimOff_times.shape[0] == contrastLeft.shape[0])
        assert(stimOff_times.shape[0] == contrastRight.shape[0])
        assert(stimOff_times.shape[0] == goCue_times.shape[0])
        assert(stimOff_times.shape[0] == choice.shape[0])
        assert(stimOff_times.shape[0] == response_times.shape[0])
        assert(stimOff_times.shape[0] == feedbackType.shape[0])
        assert(stimOff_times.shape[0] == pLeft.shape[0])
        
        nan_idx = set()
        nan_idx.update(np.where(np.isnan(stimOn_times))[0].tolist())
        nan_idx.update(np.where(np.isnan(goCue_times))[0].tolist())
        nan_idx.update(np.where(np.isnan(response_times))[0].tolist())
        nan_idx.update(np.where(np.isnan(feedback_times))[0].tolist())
        nan_idx.update(np.where(np.isnan(stimOff_times))[0].tolist())
        nan_idx.update(np.where(np.isnan(pLeft))[0].tolist())
        nan_idx = list(nan_idx)
        
        kept_idx = np.ones(stimOn_times.shape[0]).astype(bool)
        kept_idx[nan_idx] = False
        
        stimOn_times = stimOn_times[kept_idx]
        contrastLeft = contrastLeft[kept_idx]
        contrastRight = contrastRight[kept_idx]
        goCue_times = goCue_times[kept_idx]
        response_times = response_times[kept_idx]
        choice = choice[kept_idx]
        feedbackType = feedbackType[kept_idx]
        feedback_times = feedback_times[kept_idx]
        stimOff_times = stimOff_times[kept_idx]
        pLeft = pLeft[kept_idx]
        print('trial info retrieved')

    except: # relevant info not available
        return 0, 0, 0, 0, 0, False
    
    lower_limit = 0
    upper_limit = 5
    trimmed_len = 2
    # select trials with length between 0 ~ 5 seconds
    stim_diff = stimOff_times - stimOn_times
    selected = np.logical_and(stim_diff >= lower_limit, stim_diff <= upper_limit)
    stim_diff = stim_diff[selected]
    print('number of trials found: {}'.format(stim_diff.shape[0]))
    
    stimOn_times = stimOn_times[selected]
    stimOff_times = stimOff_times[selected]
    contrastLeft = contrastLeft[selected]
    contrastRight = contrastRight[selected]
    goCue_times = goCue_times[selected]
    choice = choice[selected]
    response_times = response_times[selected]
    feedbackType = feedbackType[selected]
    feedback_times = feedback_times[selected]
    pLeft = pLeft[selected]
    
    # trial_info
    trial_info = {}
    
    # padding
    pre_padding = 0.2
    post_padding = 0.
    padding_length = pre_padding + post_padding

    # n_trials
    n_trials = stim_diff.shape[0]
    trial_length = int((trimmed_len+padding_length)/bin_size)
    
    #trial_length_array = np.ceil((stim_diff + padding_length) / bin_size).astype(int)
    trial_length_array = (np.ones_like(stim_diff) * trial_length).astype(int)
    
    # get relevant dlcs
    left_dlcs = get_relevant_columns(left_dlc)
    right_dlcs = get_relevant_columns(right_dlc)
    
    # get XYs
    left_XYs = get_dlc_XYs(left_dlc.copy())
    right_XYs = get_dlc_XYs(right_dlc.copy())
    
    # get licks
    lick_times = []
    for video_type in ['right','left']:
        if video_type == 'left':
            lick_times.append(left_dlc_times[get_licks(left_XYs)])
        else:
            lick_times.append(right_dlc_times[get_licks(right_XYs)])
    lick_times = np.asarray(sorted(np.concatenate(lick_times)))
    
    # get wheel velocity
    vel = wh.velocity(wheel_timestamps,wheel_positions)
    wheel_timestamps = wheel_timestamps[~np.isnan(vel)]
    vel = vel[~np.isnan(vel)]

    probe = trajectory['probe_name']
    good_clusters = clusters[probe]['metrics']['cluster_id'][clusters[probe]['metrics']['label'] == 1].to_numpy()
    good_clusters_fr = clusters[probe]['metrics']['firing_rate'].to_numpy()[good_clusters]
    good_clusters_high_fr_idx = good_clusters_fr.argsort()[::-1][:20]
    good_clusters = good_clusters[good_clusters_high_fr_idx]
    
    good_clusters_pr = clusters[probe]['metrics']['presence_ratio'].to_numpy()[good_clusters]
    good_clusters_high_pr_idx = good_clusters_pr.argsort()[::-1][:10]
    good_clusters = good_clusters[good_clusters_high_pr_idx]
    print("good cluster id: ", good_clusters)
    
    # load spike amps + spike wf width
    amps = one.load_dataset(eid, 'clusters.amps.npy', collection='alf/{}'.format(probe))
    amps = amps[good_clusters]
    
    ptt = one.load_dataset(eid, 'clusters.peakToTrough.npy', collection='alf/{}'.format(probe))
    ptt = ptt[good_clusters]
    
    n_clusters = good_clusters.shape[0]
    n_batches = n_clusters * n_trials
    print('number of bins: {}'.format(trial_length))
    feature = np.zeros((n_clusters, n_trials, trial_length, acronym_offset+28))
    output = np.zeros((n_clusters, n_trials, trial_length))
    
    # lab specific
    feature[:,:,:,lab_offset+lab_id] = 1

    xs = clusters[probe]['x']
    ys = clusters[probe]['y']
    zs = clusters[probe]['z']
    
    # cluster specific
    for j, cluster in notebook.tqdm(enumerate(good_clusters)):
        spike_array = spikes[probe]['times'][spikes[probe]['clusters'] == cluster]
        
        # iterate through each trial
        for k in range(n_trials):
            trial_start = stimOn_times[k] - pre_padding
            trial_end = stimOn_times[k] + trimmed_len + post_padding
            # iterate through the trial
            #bin_loc = np.arange(trial_start, trial_start+upper_limit+post_padding, bin_size)
            bin_loc = np.arange(trial_start, trial_end, bin_size)
            if bin_loc.shape[0] > trial_length:
                bin_loc = bin_loc[:trial_length]
            for idx, i in enumerate(bin_loc):
                spike_num = spike_array[np.logical_and(spike_array >= i, 
                                                       spike_array < i + bin_size)].shape[0]
            
                output[j,k,idx] = spike_num / bin_size

        x = xs[cluster]
        y = ys[cluster]
        z = zs[cluster]

        # acronym
        acronym = clusters[probe]['acronym'][cluster]
        print(acronym)
        acronym_idx = acronym_dict[acronym]

        feature[j,:,:,0] = j
        
        feature[j,:,:,xyz_offset] = x
        feature[j,:,:,xyz_offset+1] = y
        feature[j,:,:,xyz_offset+2] = z
        
        feature[j,:,:,acronym_offset+acronym_idx] = 1
        
        # spike max ptp
        feature[j,:,:,max_ptp_offset] = amps[j]
        
        # spike wf width
        feature[j,:,:,wf_width_offset] = ptt[j]
        
    # trial specific
    for k in notebook.tqdm(range(n_trials)):
        stimOn_times_k = stimOn_times[k]
        stimOff_times_k = stimOff_times[k]
        trial_start = stimOn_times_k - pre_padding
        trial_end = stimOn_times_k + trimmed_len + post_padding
        stim_diff_k = stim_diff[k]
        contrastLeft_k = contrastLeft[k]
        contrastRight_k = contrastRight[k]
        goCue_times_k = goCue_times[k]
        choice_k = choice[k]
        response_times_k = response_times[k]
        feedbackType_k = feedbackType[k]
        feedback_times_k = feedback_times[k]
        pLeft_k = pLeft[k]
        
        bin_loc = np.arange(trial_start, trial_end, bin_size)
        if bin_loc.shape[0] > trial_length:
            bin_loc = bin_loc[:trial_length]
            
        for idx, i in enumerate(bin_loc):
            if idx == 0:
                left_in_range_prev = np.logical_and(left_dlc_times >= i - bin_size, 
                                               left_dlc_times < i)
                right_in_range_prev = np.logical_and(right_dlc_times >= i - bin_size, 
                                               right_dlc_times < i)
                
                f = i - bin_size
                while left_in_range_prev.astype(int).sum() == 0 or right_in_range_prev.astype(int).sum() == 0:
                    left_in_range_prev = np.logical_and(left_dlc_times >= f - bin_size, 
                                                   left_dlc_times < f)
                    right_in_range_prev = np.logical_and(right_dlc_times >= f - bin_size, 
                                                   right_dlc_times < f)
                    f -= bin_size

                left_dlc_k_prev = left_dlcs[:,left_in_range_prev]
                right_dlc_k_prev = right_dlcs[:,right_in_range_prev]

                if left_in_range_prev.astype(int).sum() > 1:
                    left_dlc_k_prev = left_dlc_k_prev.mean(1)
                else:
                    left_dlc_k_prev = np.zeros((18,1))

                if right_in_range_prev.astype(int).sum() > 1:
                    right_dlc_k_prev = right_dlc_k_prev.mean(1)
                else:
                    right_dlc_k_prev = np.zeros((18,1))
                
            left_in_range = np.logical_and(left_dlc_times >= i, 
                                           left_dlc_times < i + bin_size)
            right_in_range = np.logical_and(right_dlc_times >= i, 
                                           right_dlc_times < i + bin_size)

            left_dlc_k = left_dlcs[:,left_in_range]
            right_dlc_k = right_dlcs[:,right_in_range]

            if left_in_range.astype(int).sum() > 1:
                left_dlc_k = left_dlc_k.mean(1)
                
            if right_in_range.astype(int).sum() > 1:
                right_dlc_k = right_dlc_k.mean(1)

            if left_in_range.astype(int).sum() > 0:   
                feature[:,k,idx,left_dlc_offset:right_dlc_offset] = left_dlc_k.T - left_dlc_k_prev.T
                left_dlc_k_prev = left_dlc_k
            
            if right_in_range.astype(int).sum() > 0:   
                feature[:,k,idx,right_dlc_offset:stimOnOff_offset] = right_dlc_k.T - right_dlc_k_prev.T
                right_dlc_k_prev = right_dlc_k

            # goCue
            if np.logical_and(goCue_times >= i, goCue_times < i+bin_size).astype(int).sum() != 0:
                feature[:,k,idx,goCue_offset] = 1

            # choice
            if np.logical_and(response_times >= i, response_times < i+bin_size).astype(int).sum() != 0:
                try:
                    choice_k = choice[np.logical_and(response_times >= i, response_times < i+bin_size)]
                    chosen = choice_k + 1
                    feature[:,k,idx,choice_offset+chosen]=1
                except:
                    print('error in trial {}'.format(k))
                    pass

            # reward
            if np.logical_and(feedback_times >= i, feedback_times < i+bin_size).astype(int).sum() != 0:
                feedbackType_k = feedbackType[np.logical_and(feedback_times >= i, feedback_times < i+bin_size)]
                # reward
                if feedbackType_k == -1:
                    feature[:,k,idx,reward_offset] = 1
                else:
                    feature[:,k,idx,reward_offset+1] = 1

                    
            # wheel velocity
            w_vel_idx = np.logical_and(wheel_timestamps >= i, wheel_timestamps < i+bin_size)
            if w_vel_idx.astype(int).sum() > 1:
                feature[:,k,idx,wheel_offset] = np.mean(vel[w_vel_idx])
            elif w_vel_idx.astype(int).sum() == 1:
                feature[:,k,idx,wheel_offset] = vel[w_vel_idx]
            
            # lick
            lick_idx = lick_times[np.logical_and(lick_times >= i, lick_times < i+bin_size)]
            if lick_idx.shape[0] != 0:
                feature[:,k,idx,lick_offset] = lick_idx.shape[0]

            
        trial_offset = 0
        while k + trial_offset < stimOn_times.shape[0] and stimOn_times[k + trial_offset] < trial_end:
            cur_idx = k + trial_offset
            stimOn_bin = int(np.floor((stimOn_times[cur_idx] - trial_start) / bin_size))
            stimOff_bin = int(np.ceil((stimOff_times[cur_idx] - trial_start) / bin_size))
            if stimOff_bin > trial_length:
                stimOff_bin = trial_length
            
            contrastLeft_k = contrastLeft[cur_idx]
            contrastRight_k = contrastRight[cur_idx]
            pLeft_k = pLeft[cur_idx]
            
            feature[:,k,stimOn_bin:stimOff_bin,stimOnOff_offset] = 1
            feature[:,k,stimOn_bin:stimOff_bin,pLeft_offset] = pLeft_k
            if contrastLeft_k != 0 and not np.isnan(contrastLeft_k):
                feature[:,k,stimOn_bin:stimOff_bin,contrast_offset] = contrastLeft_k
            elif contrastRight_k != 0 and not np.isnan(contrastRight_k):
                feature[:,k,stimOn_bin:stimOff_bin,contrast_offset+1] = contrastRight_k
            
            trial_offset += 1

    return feature, output, trial_length_array, good_clusters, nan_idx, True




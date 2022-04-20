"""
functions for getting/processing features for training MTNN
"""

import numpy as np
from one.api import ONE
import sys 
sys.path.append('..')
from reproducible_ephys_functions import query, eid_list
from tqdm import notebook
import brainbox as bb
import brainbox.io.one as bbone
from collections import defaultdict
import brainbox.behavior.wheel as wh
from brainbox.behavior.dlc import get_pupil_diameter, get_smooth_pupil_diameter
from ibllib.io import spikeglx
from brainbox import singlecell
from collections import Counter
from scipy.interpolate import interp1d

from ibllib.qc.camera import CameraQC
import os

from models.expSmoothing_prevAction import expSmoothing_prevAction as exp_prevAction
from models import utils

from brainbox.population.decode import get_spike_counts_in_bins
from brainbox.task.closed_loop import compute_comparison_statistics

lab_offset = 1
session_offset = 6
xyz_offset = 10
max_ptp_offset = 13
wf_width_offset = 14
paw_offset = 15
nose_offset = 16
pupil_offset = 17
left_me_offset = 18
stimulus_offset = 19
goCue_offset = 21
firstMovement_offset = 22
choice_offset = 23
reward_offset = 25
wheel_offset = 27
pLeft_offset = 28
pLeft_last_offset = 29
lick_offset = 30
glmhmm_offset = 31 # changed to k=4 model
acronym_offset = 35
noise_offset = 40

static_idx = np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,13,14,28,29,31,32,33,34,35,36,37,38,39]) - 1
static_bool = np.zeros(noise_offset).astype(bool)
static_bool[static_idx] = True

cov_idx_dict = {'lab': (lab_offset,session_offset),
               'session': (session_offset,xyz_offset),
               'x': (xyz_offset,xyz_offset+1),
               'y': (xyz_offset+1,xyz_offset+2),
               'z': (xyz_offset+2,max_ptp_offset),
               'waveform amplitude': (max_ptp_offset,wf_width_offset),
               'waveform width': (wf_width_offset,paw_offset),
               'paw speed': (paw_offset,nose_offset),
               'nose speed': (nose_offset,pupil_offset),
               'pupil diameter': (pupil_offset,left_me_offset),
               'motion energy': (left_me_offset,stimulus_offset),
               'stimuli': (stimulus_offset,goCue_offset),
               'go cue': (goCue_offset,firstMovement_offset),
               'first movement': (firstMovement_offset,choice_offset),
               'choice': (choice_offset,reward_offset),
               'reward': (reward_offset,wheel_offset),
               'wheel velocity': (wheel_offset,pLeft_offset),
               'mouse prior': (pLeft_offset,pLeft_last_offset),
               'last mouse prior': (pLeft_last_offset,lick_offset),
               'lick': (lick_offset,glmhmm_offset),
               'decision strategy (GLM-HMM)': (glmhmm_offset,acronym_offset),
               'brain region': (acronym_offset,noise_offset),
               'noise': (noise_offset,noise_offset+1),
               'all': (1,noise_offset+1)}

glm_leave_one_out_cov_idx = (cov_idx_dict['lab'],
                             cov_idx_dict['session'],
                             cov_idx_dict['x'],
                             cov_idx_dict['y'],
                             cov_idx_dict['z'],
                             cov_idx_dict['waveform amplitude'],
                             cov_idx_dict['waveform width'],
                             cov_idx_dict['paw speed'],
                             cov_idx_dict['nose speed'],
                             cov_idx_dict['pupil diameter'],
                             cov_idx_dict['motion energy'],
                             cov_idx_dict['go cue'],
                             cov_idx_dict['choice'], 
                             cov_idx_dict['lick'],
                             cov_idx_dict['decision strategy (GLM-HMM)'],
                             cov_idx_dict['brain region'],
                             cov_idx_dict['noise'])

grouped_cov_idx_dict = {'lab': ('lab',),
                        'session': ('session',),
                        'ephys': ('x','y','z','waveform amplitude','waveform width','brain region',),
                        'task': ('stimuli','go cue','choice','reward',),
                        'behavioral': ('paw speed','nose speed','pupil diameter',
                                       'lick','motion energy','wheel velocity','first movement'),
                        'mouse prior': ('mouse prior','last mouse prior'), 
                        'decision strategy (GLM-HMM)': ('decision strategy (GLM-HMM)',),
                        'noise': ('noise',),
                        'all': ('all',)}

def check_mtnn_criteria(one=None):
    if one is None:
        one = ONE()
    
    mtnn_criteria = defaultdict(dict)
    repeated_site_eids = eid_list()
    print('total number of repeated site eids: {}'.format(len(repeated_site_eids)))
    
    for eid in repeated_site_eids:
        # check video qc
        video_qc = []
        var_sess = one.get_details(eid, full=True)
        ext_qc = var_sess['extended_qc']
        #print(ext_qc)
        criteria = [[x,ext_qc[x]] for x in ext_qc if 'dlcLeft' in x]
        for c in range(len(criteria)):
            if criteria[c][1] == False:
                video_qc.append(criteria[c][0])
        mtnn_criteria[eid]['failed video qc'] = video_qc

        has_GPIO = False
        has_ME = False
        has_passive = False
        pykilosort_available=False
        dataset_list = one.list_datasets(eid)
        for dataset in dataset_list:
            # check GPIO
            if 'leftCamera.GPIO.bin' in dataset:
                has_GPIO = True
        
            # check motion energy
            if 'leftCamera.ROIMotionEnergy.npy' in dataset:
                has_ME = True

            # check passive
            if '_ibl_passiveGabor.table.csv' in dataset:
                has_passive = True
                
            # check pykilosort
            if 'pykilosort' in dataset:
                pykilosort_available = True
                
        mtnn_criteria[eid]['GPIO'] = has_GPIO
        mtnn_criteria[eid]['ME'] = has_ME
        mtnn_criteria[eid]['PASSIVE'] = has_passive
        mtnn_criteria[eid]['PYKILOSORT'] = pykilosort_available
    
    return mtnn_criteria

def run_exp_prevAction(mtnn_eids, one=None):
    if one is None:
        one = ONE()

    for eid in notebook.tqdm(mtnn_eids.keys()):

        print('processing {}'.format(eid))
        stimuli_arr, actions_arr, stim_sides_arr, session_uuids = [], [], [], []
        data = utils.load_session(eid)
        if data['choice'] is not None and data['probabilityLeft'][0]==0.5:
            stim_side, stimuli, actions, pLeft_oracle = utils.format_data(data)
            stimuli_arr.append(stimuli)
            actions_arr.append(actions)
            stim_sides_arr.append(stim_side)
            session_uuids.append(eid)

        # format data
        stimuli, actions, stim_side = utils.format_input(stimuli_arr, actions_arr, stim_sides_arr)
        session_uuids = np.array(session_uuids)

        model = exp_prevAction('./priors/', session_uuids, eid, actions.astype(np.float32), stimuli, stim_side)
        model.load_or_train(remove_old=True)
        param = model.get_parameters() # if you want the parameters
        signals = model.compute_signal(signal=['prior', 'prediction_error', 'score'], verbose=False) # compute signals of interest

        np.save('priors/prior_{}.npy'.format(eid), signals['prior'])


def combine_regions(regions):
    """
    Combine all layers of cortex and the dentate gyrus molecular and granular layer
    Combine VISa and VISam into PPC
    """
    remove = ['1', '2', '3', '4', '5', '6a', '6b', '/']
    idx = []
    for i, region in enumerate(regions):
        if region == 'CA1':
            idx.append(i)
            continue
        if region == 'PO':
            idx.append(i)
            continue
        if region == 'LP':
            idx.append(i)
            continue
        if (region == 'DG-mo') or (region == 'DG-sg'):
            regions[i] = 'DG'
            idx.append(i)
        for j, char in enumerate(remove):
            region = region.replace(char, '')
        if (region == 'VISa') | (region == 'VISam'):
            regions[i] = 'PPC'
            idx.append(i)
    return regions, idx

def get_lab_number_map():
    lab_number_map = {'angelakilab':0, 'hoferlab': 1, 'mrsicflogellab': 1, 
                      'mainenlab': 2, 'churchlandlab': 3, 'danlab': 4}
    return lab_number_map

def get_acronym_dict():
    acronym_dict = {'LP':0, 'CA1':1, 'DG':2, 'PPC':3, 'PO':4}
    return acronym_dict

def get_acronym_dict_reverse():
    acronym_dict_reverse = {0:'LP', 1:'CA1', 2:'DG', 3:'PPC', 4:'PO'}
    return acronym_dict_reverse

def get_region_colors():
    region_colors = {'LP':'k', 'CA1':'b', 'DG':'r', 'PPC':'g', 'PO':'y'}
    return region_colors

def get_mtnn_eids():
#     mtnn_eids = {'56b57c38-2699-4091-90a8-aba35103155e': 0,
#                 '72cb5550-43b4-4ef0-add5-e4adfdfb5e02': 0,
# #                 '746d1902-fa59-4cab-b0aa-013be36060d5': 2,
#                 'dac3a4c1-b666-4de0-87e8-8c514483cacf': 0,
#                 '6f09ba7e-e3ce-44b0-932b-c003fb44fb89': 0,
#                 'f312aaec-3b6f-44b3-86b4-3a0c119c0438': 3,
#                 'dda5fc59-f09a-4256-9fb5-66c67667a466': 2,
#                 'ee40aece-cffd-4edb-a4b6-155f158c666a': 3,
#                 'ecb5520d-1358-434c-95ec-93687ecd1396': 3,
#                 '54238fd6-d2d0-4408-b1a9-d19d24fd29ce': 0,
#                 '51e53aff-1d5d-4182-a684-aba783d50ae5': 0,
#                 'c51f34d8-42f6-4c9c-bb5b-669fd9c42cd9': 0,
# #                 '0802ced5-33a3-405e-8336-b65ebc5cb07c': 0,
#                 'db4df448-e449-4a6f-a0e7-288711e7a75a': 2,
#                 '30c4e2ab-dffc-499d-aae4-e51d6b3218c2': 0,
#                 '4a45c8ba-db6f-4f11-9403-56e06a33dfa4': 0,
#                 '7af49c00-63dd-4fed-b2e0-1b3bd945b20b': 0,
#                 '4b00df29-3769-43be-bb40-128b1cba6d35': 0,
#                 'f140a2ec-fd49-4814-994a-fe3476f14e66': 0,
#                 '862ade13-53cd-4221-a3fa-dda8643641f2': 0,
#                 'c7248e09-8c0d-40f2-9eb4-700a8973d8c8': 1,
# #                 '88224abb-5746-431f-9c17-17d7ef806e6a': 0,
# #                 'd0ea3148-948d-4817-94f8-dcaf2342bbbe': 0,
#                 'd23a44ef-1402-4ed7-97f5-47e9a7a504d9': 0}

    mtnn_eids = {'51e53aff-1d5d-4182-a684-aba783d50ae5': 0,
                 'c51f34d8-42f6-4c9c-bb5b-669fd9c42cd9': 0,
                 '7af49c00-63dd-4fed-b2e0-1b3bd945b20b': 0,
                 'f140a2ec-fd49-4814-994a-fe3476f14e66': 0,
                 '56b57c38-2699-4091-90a8-aba35103155e': 0,
                 'dac3a4c1-b666-4de0-87e8-8c514483cacf': 0,
                 '6f09ba7e-e3ce-44b0-932b-c003fb44fb89': 0,
                 '862ade13-53cd-4221-a3fa-dda8643641f2': 0,
                 '72cb5550-43b4-4ef0-add5-e4adfdfb5e02': 0,
                 'ee40aece-cffd-4edb-a4b6-155f158c666a': 3,
                 '30c4e2ab-dffc-499d-aae4-e51d6b3218c2': 0,
                 'c7248e09-8c0d-40f2-9eb4-700a8973d8c8': 1,
                 'f312aaec-3b6f-44b3-86b4-3a0c119c0438': 3,
                 'dda5fc59-f09a-4256-9fb5-66c67667a466': 2,
                 'ecb5520d-1358-434c-95ec-93687ecd1396': 3,
                 '4b00df29-3769-43be-bb40-128b1cba6d35': 0,
                 '54238fd6-d2d0-4408-b1a9-d19d24fd29ce': 0,
                 'db4df448-e449-4a6f-a0e7-288711e7a75a': 2,
                 '4a45c8ba-db6f-4f11-9403-56e06a33dfa4': 0,
                 'd23a44ef-1402-4ed7-97f5-47e9a7a504d9': 0,
                }
    
    return mtnn_eids

def get_traj(eids):
    traj = query()
    tmp = []
    for eid in eids:
        for t in traj:
            if t['session']['id'] == eid:
                tmp.append(t)
                break
    traj = tmp
    
    return traj


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
            dlc[point + '_likelihood'] < 0., dlc[point + '_x']) # don't fill with nans
        x = x.filled(np.nan)
        y = np.ma.masked_where(
            dlc[point + '_likelihood'] < 0., dlc[point + '_y']) # don't fill with nans
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
def featurize(i, trajectory, one, session_counter, 
              bin_size=0.05, align_event='movement_onset',
              t_before=0.5, t_after=1.0):
    lab_number_map = get_lab_number_map()
    acronym_dict = get_acronym_dict()
    mtnn_eids = get_mtnn_eids()

    eid = trajectory['session']['id']
    subject = trajectory['session']['subject']
    probe = trajectory['probe_name']
    date_number = trajectory['session']['start_time'].split('T')[0]+'-00'+str(trajectory['session']['number'])
    print('processing {}: {}'.format(subject, eid))
    lab_id = lab_number_map[trajectory['session']['lab']]
    for key in session_counter.keys():
        if trajectory['session']['lab'] in key:
            session_id = session_counter[key]
            session_counter[key] += 1
#     spikes = one.load_object(eid, 'spikes', collection='alf/{}/pykilosort'.format(probe))
#     clusters = one.load_object(eid, 'clusters', collection='alf/{}/pykilosort'.format(probe))
#     channels = one.load_object(eid, 'channels', collection='alf/{}/pykilosort'.format(probe))
#     clusters = bbone.merge_clusters_channels(dic_clus={probe: clusters}, channels=channels)[probe]
    spikes, clusters, channels = bbone.load_spike_sorting_with_channel(eid, 
                                                                     one=one, 
                                                                     probe=probe, 
                                                                     spike_sorter='pykilosort')
    spikes = spikes[probe]
    clusters = clusters[probe]
    channels = channels[probe]
    #print('spikes retrieved')

    #print('loading motion energy and dlc')
    left_me = one.load_dataset(eid, 'leftCamera.ROIMotionEnergy.npy')
    left_dlc = one.load_dataset(eid, '_ibl_leftCamera.dlc.pqt')
    left_dlc_times = one.load_dataset(eid, '_ibl_leftCamera.times.npy')
    if left_dlc_times.shape[0] != left_dlc.shape[0]:
        left_offset = mtnn_eids[eid]
        left_dlc_times = left_dlc_times[abs(left_offset):abs(left_offset)+left_dlc.shape[0]]
    assert(left_dlc_times.shape[0] == left_dlc.shape[0])
    #print('motion energy + dlc retrieved')

    stimOn_times = one.load_dataset(eid, '_ibl_trials.stimOn_times.npy')
    trial_numbers = np.arange(stimOn_times.shape[0])
    firstMovement_times = one.load_dataset(eid, '_ibl_trials.firstMovement_times.npy')

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

    # load pLeft (Charles's model's output)
    #print('loading pLeft')
    pLeft = np.load('./priors/prior_{}.npy'.format(eid))

    #print('loading glm hmm')
    glm_hmm = np.load('glm_hmm/k=4/posterior_probs_valuessession_{}-{}.npz'.format(subject,date_number))
    for item in glm_hmm.files:
        glm_hmm_states = glm_hmm[item]

    # filter out trials with no choice
    choice_filter = np.where(choice!=0)

    stimOn_times = stimOn_times[choice_filter]
    firstMovement_times = firstMovement_times[choice_filter]
    contrastLeft = contrastLeft[choice_filter]
    contrastRight = contrastRight[choice_filter]
    goCue_times = goCue_times[choice_filter]
    response_times = response_times[choice_filter]
    choice = choice[choice_filter]
    feedbackType = feedbackType[choice_filter]
    feedback_times = feedback_times[choice_filter]
    stimOff_times = stimOff_times[choice_filter]
    pLeft = pLeft[choice_filter]
    trial_numbers = trial_numbers[choice_filter]
    
    # filter out trials with no contrast
    contrast_filter1 = np.logical_and(np.isnan(contrastLeft), contrastRight==0)
    contrast_filter2 = np.logical_and(contrastLeft==0, np.isnan(contrastRight))
    contrast_filter = ~np.logical_or(contrast_filter1,contrast_filter2)
    
    stimOn_times = stimOn_times[contrast_filter]
    firstMovement_times = firstMovement_times[contrast_filter]
    contrastLeft = contrastLeft[contrast_filter]
    contrastRight = contrastRight[contrast_filter]
    goCue_times = goCue_times[contrast_filter]
    response_times = response_times[contrast_filter]
    choice = choice[contrast_filter]
    feedbackType = feedbackType[contrast_filter]
    feedback_times = feedback_times[contrast_filter]
    stimOff_times = stimOff_times[contrast_filter]
    pLeft = pLeft[contrast_filter]
    trial_numbers = trial_numbers[contrast_filter]
    glm_hmm_states = glm_hmm_states[contrast_filter]

    assert(stimOff_times.shape[0] == feedback_times.shape[0])
    assert(stimOff_times.shape[0] == firstMovement_times.shape[0])
    assert(stimOff_times.shape[0] == contrastLeft.shape[0])
    assert(stimOff_times.shape[0] == contrastRight.shape[0])
    assert(stimOff_times.shape[0] == goCue_times.shape[0])
    assert(stimOff_times.shape[0] == choice.shape[0])
    assert(stimOff_times.shape[0] == response_times.shape[0])
    assert(stimOff_times.shape[0] == feedbackType.shape[0])
    assert(stimOff_times.shape[0] == pLeft.shape[0])

    nan_idx = set()
    nan_idx.update(np.where(np.isnan(stimOn_times))[0].tolist())
    nan_idx.update(np.where(np.isnan(firstMovement_times))[0].tolist())
    nan_idx.update(np.where(np.isnan(goCue_times))[0].tolist())
    nan_idx.update(np.where(np.isnan(response_times))[0].tolist())
    nan_idx.update(np.where(np.isnan(feedback_times))[0].tolist())
    nan_idx.update(np.where(np.isnan(stimOff_times))[0].tolist())
    nan_idx.update(np.where(np.isnan(pLeft))[0].tolist())
    nan_idx = list(nan_idx)

    kept_idx = np.ones(stimOn_times.shape[0]).astype(bool)
    kept_idx[nan_idx] = False

    stimOn_times = stimOn_times[kept_idx]
    firstMovement_times = firstMovement_times[kept_idx]
    contrastLeft = contrastLeft[kept_idx]
    contrastRight = contrastRight[kept_idx]
    goCue_times = goCue_times[kept_idx]
    response_times = response_times[kept_idx]
    choice = choice[kept_idx]
    feedbackType = feedbackType[kept_idx]
    feedback_times = feedback_times[kept_idx]
    stimOff_times = stimOff_times[kept_idx]
    pLeft = pLeft[kept_idx]
    glm_hmm_states = glm_hmm_states[kept_idx]
    trial_numbers = trial_numbers[kept_idx]
    #print('trial info retrieved')

    # select trials
    if align_event == 'movement_onset':
        ref_event = firstMovement_times
        diff1 = ref_event - stimOn_times
        diff2 = feedback_times - ref_event
        t_select1 = np.logical_and(diff1 > 0.0, diff1 < t_before-0.1)
        t_select2 = np.logical_and(diff2 > 0.0, diff2 < t_after-0.1)
        t_select = np.logical_and(t_select1, t_select2)
        
    stimOn_times = stimOn_times[t_select]
    firstMovement_times = firstMovement_times[t_select]
    contrastLeft = contrastLeft[t_select]
    contrastRight = contrastRight[t_select]
    goCue_times = goCue_times[t_select]
    response_times = response_times[t_select]
    choice = choice[t_select]
    feedbackType = feedbackType[t_select]
    feedback_times = feedback_times[t_select]
    stimOff_times = stimOff_times[t_select]
    pLeft = pLeft[t_select]
    pLeft_last = np.roll(pLeft, 1)
    glm_hmm_states = glm_hmm_states[t_select]
    trial_numbers = trial_numbers[t_select]
    ref_event = ref_event[t_select]
        
    n_active_trials = ref_event.shape[0]
    
    # n_trials
    n_trials = n_active_trials
    print('number of trials found: {} (active: {})'.format(n_trials,n_active_trials))
    
    # get XYs
    left_XYs = get_dlc_XYs(left_dlc.copy())

    #print('getting lick times')
    # get licks
    lick_times = []
    lick_times.append(left_dlc_times[get_licks(left_XYs)])
    lick_times = np.asarray(sorted(np.concatenate(lick_times)))
    
    fs = 60
    #print('getting paw speed')
    # get right paw speed (closer to camera)
    x = left_XYs['paw_r'][0]/2
    y = left_XYs['paw_r'][1]/2 

    # get speed in px/sec [half res]
    paw_speed = ((np.diff(x)**2 + np.diff(y)**2)**.5)*fs
    paw_speed = np.append(paw_speed, paw_speed[-1])
    
    #print('getting nose speed')
    # get nose speed
    x = left_XYs['nose_tip'][0]/2
    y = left_XYs['nose_tip'][1]/2 

    # get speed in px/sec [half res]
    nose_speed = ((np.diff(x)**2 + np.diff(y)**2)**.5)*fs
    nose_speed = np.append(nose_speed, nose_speed[-1])
    
    #print('getting pupil diameter')
    # get pupil diameter
    #pupil_diameter = get_pupil_diameter(left_XYs)
    raw_pupil_diameter = get_pupil_diameter(left_dlc.copy())
    pupil_diameter = get_smooth_pupil_diameter(raw_pupil_diameter, 'left')
    
    #print('getting wheel velocity')
    # get wheel velocity
    vel = wh.velocity(wheel_timestamps,wheel_positions)
    wheel_timestamps = wheel_timestamps[~np.isnan(vel)]
    vel = vel[~np.isnan(vel)]

    good_clusters = clusters['metrics']['cluster_id'][clusters['metrics']['label'] == 1].to_numpy()
    acronyms = clusters['acronym'][good_clusters]
    rs_regions, idx = combine_regions(acronyms)
    rs_regions = rs_regions[idx]
    good_clusters = good_clusters[idx]
    print("repeated site brain region counts: ", Counter(rs_regions))
    print("number of good clusters: ", good_clusters.shape[0])
    
    good_spike_idx = np.isin(spikes['clusters'], good_clusters)
    spk_times = spikes['times'][good_spike_idx]
    spk_clusters = spikes['clusters'][good_spike_idx]
#     significant_units, _, _, _ = responsive_units(spk_times,spk_clusters,firstMovement_times, 
#                                              pre_time=[0.2,0], post_time=[0.05,0.2], use_fr=True)
#     responsive_idx = np.isin(good_clusters, significant_units)
#     rs_regions = rs_regions[responsive_idx]
#     good_clusters = good_clusters[responsive_idx]

    # Find responsive neurons
    cluster_ids = np.unique(spk_clusters)
    # Baseline firing rate
    intervals = np.c_[stimOn_times - 0.2, stimOn_times]
    counts, cluster_ids = get_spike_counts_in_bins(spk_times, spk_clusters, intervals)
    fr_base = counts / (intervals[:, 1] - intervals[:, 0])

    # Post-move firing rate
    intervals = np.c_[firstMovement_times - 0.05, firstMovement_times + 0.2]
    counts, cluster_ids = get_spike_counts_in_bins(spk_times, spk_clusters, intervals)
    fr_post_move = counts / (intervals[:, 1] - intervals[:, 0])

    sig_units, _, _ = compute_comparison_statistics(fr_base, fr_post_move, test='signrank')
    significant_units = cluster_ids[sig_units]
    responsive_idx = np.isin(good_clusters, significant_units)
    rs_regions = rs_regions[responsive_idx]
    good_clusters = good_clusters[responsive_idx]
    print("(responsive) repeated site brain region counts: ", Counter(rs_regions))
    print("(responsive) number of good clusters: ", good_clusters.shape[0])

    # skip this part to get all clusters. we can filter for better clusters later on
#     good_clusters_fr = clusters['metrics']['firing_rate'].to_numpy()[good_clusters]
#     print(good_clusters_fr)
#     good_clusters_high_fr_idx = good_clusters_fr.argsort()[::-1][:40]
#     good_clusters = good_clusters[good_clusters_high_fr_idx]
#     rs_regions = rs_regions[good_clusters_high_fr_idx]
#     good_clusters_pr = clusters['metrics']['presence_ratio'].to_numpy()[good_clusters]
#     good_clusters_high_pr_idx = good_clusters_pr.argsort()[::-1][:10]
#     good_clusters = good_clusters[good_clusters_high_pr_idx]
#     rs_regions = rs_regions[good_clusters_high_pr_idx]
#     print("good cluster id: ", good_clusters)
#     print("selected brain region counts: ", Counter(rs_regions))
    
    # load spike amps + spike wf width
    amps = one.load_dataset(eid, 'clusters.amps.npy', collection='alf/{}/pykilosort'.format(probe))
    ptt = one.load_dataset(eid, 'clusters.peakToTrough.npy', collection='alf/{}/pykilosort'.format(probe))
    
    n_clusters = good_clusters.shape[0]
    n_tbins = int((t_after+t_before)/bin_size)
    feature = np.zeros((n_clusters, n_trials, n_tbins, noise_offset+1))
    output = np.zeros((n_clusters, n_trials, n_tbins))
    print("feature tensor size: {}".format(feature.shape))
    
    # session specific
    feature[:,:,:,session_offset+session_id] = 1

    # lab specific
    feature[:,:,:,lab_offset+lab_id] = 1
    
    # cluster xyz location
    xs = clusters['x']
    ys = clusters['y']
    zs = clusters['z']
    
    # cluster specific
    for j, cluster in notebook.tqdm(enumerate(good_clusters)):
        spike_array = spikes['times'][spikes['clusters'] == cluster]
        
        # iterate through each trial (active)
        for k in range(n_active_trials):
            ref_t = ref_event[k]
            start_t = ref_t-t_before
            bin_loc = start_t+np.arange(n_tbins)*bin_size
            for idx, i in enumerate(bin_loc):
                spike_num = spike_array[np.logical_and(spike_array >= i, 
                                                       spike_array < i + bin_size)].shape[0]
            
                output[j,k,idx] = spike_num / bin_size

        x = xs[cluster]
        y = ys[cluster]
        z = zs[cluster]

        # acronym
        acronym = rs_regions[j]
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
    for k in notebook.tqdm(range(n_active_trials)):
        stimOn_times_k = stimOn_times[k]
        stimOff_times_k = stimOff_times[k]
        contrastLeft_k = contrastLeft[k]
        contrastRight_k = contrastRight[k]
        goCue_times_k = goCue_times[k]
        firstMovement_times_k = firstMovement_times[k]
        choice_k = choice[k]
        response_times_k = response_times[k]
        feedbackType_k = feedbackType[k]
        feedback_times_k = feedback_times[k]
        pLeft_k = pLeft[k]
        pLeft_last_k = pLeft_last[k]
        glm_hmm_states_k = glm_hmm_states[k]
        
        ref_t = ref_event[k]
        start_t = ref_t-t_before
        bin_loc = start_t+np.arange(n_tbins)*bin_size
        for idx, i in enumerate(bin_loc):
            left_in_range = np.logical_and(left_dlc_times >= i, 
                                           left_dlc_times < i + bin_size)

            left_me_k = left_me[left_in_range]
            paw_speed_k = paw_speed[left_in_range]
            nose_speed_k = nose_speed[left_in_range]
            pupil_diameter_k = pupil_diameter[left_in_range]

            if left_in_range.astype(int).sum() > 1:
                left_me_k = left_me_k.mean()
                paw_speed_k = paw_speed_k.mean()
                nose_speed_k = nose_speed_k.mean()
                pupil_diameter_k = pupil_diameter_k.mean()

            if left_in_range.astype(int).sum() > 0:
                feature[:,k,idx,paw_offset] = paw_speed_k
                feature[:,k,idx,nose_offset] = nose_speed_k
                feature[:,k,idx,pupil_offset] = pupil_diameter_k
                feature[:,k,idx,left_me_offset] = left_me_k
            else:
                feature[:,k,idx,paw_offset] = feature[:,k,idx-1,paw_offset]
                feature[:,k,idx,nose_offset] = feature[:,k,idx-1,nose_offset]
                feature[:,k,idx,pupil_offset] = feature[:,k,idx-1,pupil_offset]
                feature[:,k,idx,left_me_offset] = feature[:,k,idx-1,left_me_offset]

            # goCue
            if np.logical_and(goCue_times >= i, goCue_times < i+bin_size).astype(int).sum() != 0:
                feature[:,k,idx,goCue_offset] = 1
                
            # firstMovement
            if np.logical_and(firstMovement_times >= i, firstMovement_times < i+bin_size).astype(int).sum() != 0:
                feature[:,k,idx,firstMovement_offset] = 1

            # choice
            if np.logical_and(response_times >= i, response_times < i+bin_size).astype(int).sum() != 0:
                choice_k = choice[np.logical_and(response_times >= i, response_times < i+bin_size)]
                if choice_k == -1:
                    feature[:,k,idx,choice_offset]=1
                elif choice_k == 1:
                    feature[:,k,idx,choice_offset+1]=1

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
                feature[:,k,idx,lick_offset] = 1 #lick_idx.shape[0]
            
        # mouse prior
        feature[:,k,:,pLeft_offset] = pLeft_k
        feature[:,k,:,pLeft_last_offset] = pLeft_last_k
        
        # decision strategy
        feature[:,k,:,glmhmm_offset:acronym_offset] = glm_hmm_states_k
        
        # stimulus
        stimOn_bin = np.floor((stimOn_times_k - start_t) / bin_size).astype(int)
        if contrastLeft_k != 0 and not np.isnan(contrastLeft_k):
            feature[:,k,stimOn_bin,stimulus_offset] = contrastLeft_k
        elif contrastRight_k != 0 and not np.isnan(contrastRight_k):
            feature[:,k,stimOn_bin,stimulus_offset+1] = contrastRight_k

    return feature, output, good_clusters, trial_numbers


def reshape_flattened(flattened, shape, trim=0):
    reshaped = []
    idx = 0
    for sh in shape:
        if trim > 0:
            sh2 = sh[:-trim]
        else:
            sh2 = sh
        n = np.prod(np.asarray(sh2))
        reshaped.append(flattened[idx:idx+n].reshape(sh2))
        idx += n
    
    return reshaped

def load_original(eids):
    feature_list, output_list, cluster_number_list, session_list, trial_number_list = [], [], [], [], []
    for eid in eids:
        feature_list.append(np.load(f'./original_data/{eid}_feature.npy'))
        output_list.append(np.load(f'./original_data/{eid}_output.npy'))
        cluster_number_list.append(np.load(f'./original_data/{eid}_clusters.npy'))
        session_list.append(np.load(f'./original_data/{eid}_session_info.npy', allow_pickle=True))
        trial_number_list.append(np.load(f'./original_data/{eid}_trials.npy'))
        
    return feature_list, output_list, cluster_number_list, session_list, trial_number_list

def compute_mean_frs(shape_path='mtnn_data/train/shape.npy', obs_path='mtnn_data/train/output.npy'):
    
    shape = np.load(shape_path)
    obs = np.load(obs_path)
    
    mean_fr_list = []
    idx = 0
    for sh in shape:
        n = sh[0]*sh[1]
        mean_fr_list.extend(list(obs[idx:idx+n].reshape(sh[:-1]).mean(1).mean(1)))
        idx += n
    
    return np.asarray(mean_fr_list)
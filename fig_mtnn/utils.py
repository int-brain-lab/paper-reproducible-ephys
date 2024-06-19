"""
functions for getting/processing features for training MTNN
"""

import numpy as np
import pandas as pd
from one.api import ONE
from one.alf.exceptions import ALFObjectNotFound
import sys 
sys.path.append('..')
from reproducible_ephys_functions import get_insertions, query, traj_list_to_dataframe#, eid_list
from tqdm import notebook
from collections import defaultdict
import brainbox.behavior.wheel as wh
from collections import Counter
from scipy.interpolate import interp1d

from ibllib.qc.camera import CameraQC
import os

from fig_mtnn.models.expSmoothing_prevAction import expSmoothing_prevAction as exp_prevAction
from fig_mtnn.models import utils as mutils

from brainbox.population.decode import get_spike_counts_in_bins
from brainbox.task.closed_loop import compute_comparison_statistics
from brainbox.core import TimeSeries
from brainbox.processing import sync

from brainbox.io.one import SpikeSortingLoader
from reproducible_ephys_functions import combine_regions, BRAIN_REGIONS, repo_path, save_data_path
import brainbox.behavior.dlc as dlc
from reproducible_ephys_processing import bin_spikes, bin_spikes2D,  bin_norm, compute_new_label
from iblatlas.atlas import AllenAtlas

rng = np.random.default_rng(seed=12345)

lab_offset = 1
session_offset = 9
xyz_offset = 13
max_ptp_offset = 16
wf_width_offset = 17
paw_offset = 18
nose_offset = 19
pupil_offset = 20
left_me_offset = 21
stimulus_offset = 22
goCue_offset = 24
firstMovement_offset = 25
choice_offset = 26
reward_offset = 28
wheel_offset = 30
pLeft_offset = 31
pLeft_last_offset = 32
lick_offset = 33
glmhmm_offset = 34 # changed to k=4 model
acronym_offset = 38
noise_offset = 43

static_idx = np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,31,32,34,35,36,37,38,39,40,41,42]) - 1
static_bool = np.zeros(noise_offset).astype(bool)
static_bool[static_idx] = True

cov_idx_dict = {'lab': (lab_offset,xyz_offset), #(lab_offset,session_offset),
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
#                'all': (1,noise_offset+1)
               }

#leave_out_covs_for_glm = ['lab','session','x','y','z',
#                         'waveform amplitude','waveform width',
#                         'paw speed','nose speed','pupil diameter',
#                         'motion energy','go cue','choice','lick',
#                         'decision strategy (GLM-HMM)','brain region','noise']

leave_out_covs_for_glm = ['lab','session','x','y','z',
                         'waveform amplitude','waveform width',
                         'decision strategy (GLM-HMM)','brain region','noise']

grouped_cov_idx_dict = {'ephys': ['x','y','z','waveform amplitude','waveform width','brain region'],
                        'task': ['stimuli','go cue','first movement','choice','reward'],
                        'behavioral': ['paw speed','nose speed','pupil diameter',
                                       'lick','motion energy','wheel velocity']}

sim_cov_idx_dict = {'left stimuli': (1,2),
                    'right stimuli': (2,3),
                    'incorrect': (3,4),
                    'correct': (4,5),
                    'first movement': (5,6),
                    'mouse prior': (6,7),
                    'last mouse prior': (7,8),
                    'wheel velocity': (8,9),}

sim_static_idx = np.asarray([6,7])
sim_static_bool = np.zeros(8).astype(bool)
sim_static_bool[sim_static_idx] = True


def check_mtnn_criteria(one=None):
    if one is None:
        one = ONE()
    
    mtnn_criteria = defaultdict(dict)
    repeated_site_trajs = get_insertions(level=2, freeze='freeze_2024_03')
    repeated_site_trajs = traj_list_to_dataframe(repeated_site_trajs)
    repeated_site_eids = list(repeated_site_trajs.eid)
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

    for eid in notebook.tqdm(mtnn_eids):

        print('processing {}'.format(eid))
        stimuli_arr, actions_arr, stim_sides_arr, session_uuids = [], [], [], []
        data = mutils.load_session(eid)
        if data['choice'] is not None and data['probabilityLeft'][0]==0.5:
            stim_side, stimuli, actions, pLeft_oracle = mutils.format_data(data)
            stimuli_arr.append(stimuli)
            actions_arr.append(actions)
            stim_sides_arr.append(stim_side)
            session_uuids.append(eid)

        # format data
        stimuli, actions, stim_side = mutils.format_input(stimuli_arr, actions_arr, stim_sides_arr)
        session_uuids = np.array(session_uuids)

        model = exp_prevAction('./priors/', session_uuids, eid, actions.astype(np.float32), stimuli, stim_side)
        model.load_or_train(remove_old=True)
        param = model.get_parameters() # if you want the parameters
        signals = model.compute_signal(signal=['prior', 'prediction_error', 'score'], verbose=False) # compute signals of interest

        np.save('priors/prior_{}.npy'.format(eid), signals['prior'])


def get_lab_number_map():
    lab_number_map = {'hoferlab': 0, 'mrsicflogellab': 0, 'mainenlab': 1, 
                      'churchlandlab': 2, 'cortexlab': 3, 'danlab': 4,
                      'angelakilab': 5, 'churchlandlab_ucla': 6, 'steinmetzlab': 7}
    return lab_number_map

def get_acronym_dict():
    acronym_dict = {'LP': 0, 'CA1': 1, 'DG': 2, 'PPC': 3, 'PO': 4}
    return acronym_dict

def get_acronym_dict_reverse():
    acronym_dict_reverse = {0: 'LP', 1: 'CA1', 2: 'DG', 3: 'PPC', 4: 'PO'}
    return acronym_dict_reverse

def get_region_colors():
    region_colors = {'LP': 'k', 'CA1': 'b', 'DG': 'r', 'PPC': 'g', 'PO': 'y'}
    return region_colors

def clusters_to_skip_dict():
    skip_dict = {'6f09ba7e-e3ce-44b0-932b-c003fb44fb89': [298, 736], 
                 'dac3a4c1-b666-4de0-87e8-8c514483cacf': [482, 530], 
                 '56b57c38-2699-4091-90a8-aba35103155e': [1580], 
                 '3638d102-e8b6-4230-8742-e548cd87a949': [457, 508, 531, 591, 593], 
                 'a4a74102-2af5-45dc-9e41-ef7f5aed88be': [509, 1256], 
                 '88224abb-5746-431f-9c17-17d7ef806e6a': [1353], 
                 'c7248e09-8c0d-40f2-9eb4-700a8973d8c8': [1222, 1291, 1314, 1605, 1697], 
                 '4b7fbad4-f6de-43b4-9b15-c7c7ef44db4b': [1561], 
                 'f312aaec-3b6f-44b3-86b4-3a0c119c0438': [1181], 
                 'd04feec7-d0b7-4f35-af89-0232dd975bf0': [472, 808, 822, 854, 1190, 1481], 
                 '8928f98a-b411-497e-aa4b-aa752434686d': [1118, 1364, 1458, 1579, 1584, 1585, 1620], 
                 'db4df448-e449-4a6f-a0e7-288711e7a75a': [1217, 1533], 
                 '54238fd6-d2d0-4408-b1a9-d19d24fd29ce': [1192], 
                 'b03fbc44-3d8e-4a6c-8a50-5ea3498568e0': [843, 1305, 1325, 1445, 1568], 
                 '51e53aff-1d5d-4182-a684-aba783d50ae5': [1557], 
                 'a8a8af78-16de-4841-ab07-fde4b5281a03': [1375], 
                 '61e11a11-ab65-48fb-ae08-3cb80662e5d6': [254, 1558, 1566], 
                 'ebce500b-c530-47de-8cb1-963c552703ea': [544, 1593], 
                 '6899a67d-2e53-4215-a52a-c7021b5da5d4': [399], 
                 '15b69921-d471-4ded-8814-2adad954bcd8': [1353], 
                 '3bcb81b4-d9ca-4fc9-a1cd-353a966239ca': [1350], 
                 'f115196e-8dfe-4d2a-8af3-8206d93c1729': [1086, 1380, 1530]}
    
    return skip_dict

def get_mtnn_eids():

    eids = [
        '6f09ba7e-e3ce-44b0-932b-c003fb44fb89',  # SWC(H) *
        'dac3a4c1-b666-4de0-87e8-8c514483cacf',  # SWC(H) *
        '56b57c38-2699-4091-90a8-aba35103155e',  # SWC(M) *
        '3638d102-e8b6-4230-8742-e548cd87a949',  # SWC(M)
        
        'a4a74102-2af5-45dc-9e41-ef7f5aed88be',  # CCU *
        '88224abb-5746-431f-9c17-17d7ef806e6a',  # CCU *
        '72cb5550-43b4-4ef0-add5-e4adfdfb5e02',  # CCU *
        'c7248e09-8c0d-40f2-9eb4-700a8973d8c8',  # CCU *
        
        'dda5fc59-f09a-4256-9fb5-66c67667a466',  # CSHL(C)
        '4b7fbad4-f6de-43b4-9b15-c7c7ef44db4b',  # CSHL(C)
        'f312aaec-3b6f-44b3-86b4-3a0c119c0438',  # CSHL(C)
        '4b00df29-3769-43be-bb40-128b1cba6d35',  # CSHL(C)

        '3f859b5c-e73a-4044-b49e-34bb81e96715',  # UCL *
        'd04feec7-d0b7-4f35-af89-0232dd975bf0',  # UCL *
        'c7bf2d49-4937-4597-b307-9f39cb1c7b16',  # UCL *
        '8928f98a-b411-497e-aa4b-aa752434686d',  # UCL *

        'db4df448-e449-4a6f-a0e7-288711e7a75a',  # Berkeley
        'd23a44ef-1402-4ed7-97f5-47e9a7a504d9',  # Berkeley
        '54238fd6-d2d0-4408-b1a9-d19d24fd29ce',  # Berkeley
        'b03fbc44-3d8e-4a6c-8a50-5ea3498568e0',  # Berkeley
        
        '51e53aff-1d5d-4182-a684-aba783d50ae5',  # NYU *
        'f140a2ec-fd49-4814-994a-fe3476f14e66',  # NYU
        'a8a8af78-16de-4841-ab07-fde4b5281a03',  # NYU
        '61e11a11-ab65-48fb-ae08-3cb80662e5d6',  # NYU *

        'ebce500b-c530-47de-8cb1-963c552703ea',  # UCLA *
        '6899a67d-2e53-4215-a52a-c7021b5da5d4',  # UCLA *
        '15b69921-d471-4ded-8814-2adad954bcd8',  # UCLA *
        '824cf03d-4012-4ab1-b499-c83a92c5589e',  # UCLA *
        
        '3bcb81b4-d9ca-4fc9-a1cd-353a966239ca',  # UW
        'f115196e-8dfe-4d2a-8af3-8206d93c1729',  # UW
        '9b528ad0-4599-4a55-9148-96cc1d93fb24',  # UW
        '3e6a97d3-3991-49e2-b346-6948cb4580fb',  # UW
    ]
    
    return eids

def get_traj(eids):
    traj = get_insertions(level=2, freeze='freeze_2024_03')
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
def featurize(i, trajectory, one, session_counter, bin_size=0.05, align_event='movement_onset', t_before=0.5, t_after=1.0,
              brain_atlas=None):

    lab_number_map = get_lab_number_map()
    acronym_dict = get_acronym_dict()
    mtnn_eids = get_mtnn_eids()
    clusters_to_skip_dict_all = clusters_to_skip_dict()
 
    eid = trajectory['session']['id']
    try:
        clusters_to_skip = clusters_to_skip_dict_all[eid]
    except:
        clusters_to_skip = []
    subject = trajectory['session']['subject']
    probe = trajectory['probe_name']
    date_number = trajectory['session']['start_time'].split('T')[0]+'-00'+str(trajectory['session']['number'])
    print('processing {}: {}'.format(subject, eid))
    lab_id = lab_number_map[trajectory['session']['lab']]
    for key in session_counter.keys():
        if trajectory['session']['lab'] in key:
            session_id = session_counter[key]
            session_counter[key] += 1

    ba = brain_atlas or AllenAtlas()
    sl = SpikeSortingLoader(eid=eid, pname=probe, one=one, atlas=ba)
    spikes, clusters, channels = sl.load_spike_sorting(dataset_types=['clusters.amps', 'clusters.peakToTrough'],
                                                       revision='2024-03-22', enforce_version=False)
    clusters = sl.merge_clusters(spikes, clusters, channels)

    clusters['rep_site_acronym'] = combine_regions(clusters['acronym'])

    # Find clusters that are in the repeated site brain regions and that have been labelled as good
    cluster_idx = np.sort(np.where(np.bitwise_and(np.isin(clusters['rep_site_acronym'], BRAIN_REGIONS), clusters['label'] == 1))[0])
    cluster_idx = cluster_idx[~np.isin(cluster_idx, clusters_to_skip)]
    cluster_id = clusters['cluster_id'][cluster_idx]
    # Find the index of spikes that belong to the chosen clusters
    spike_idx = np.isin(spikes['clusters'], cluster_idx)

    print("repeated site brain region counts: ", Counter(clusters['rep_site_acronym'][cluster_idx]))
    print("number of good clusters: ", cluster_idx.shape[0])

    # Load in trials
    trials = one.load_object(eid, 'trials')
    trial_numbers = np.arange(trials['stimOn_times'].shape[0])

    # load pLeft (Charles's model's output)
    pLeft = np.load(save_data_path(figure='fig_mtnn').joinpath('priors', f'prior_{eid}.npy'))

    # load in GLM-HMM
    glm_hmm = pd.read_parquet(save_data_path(figure='fig_mtnn').joinpath('glm_hmm', 'k=4',
                                                                   f'{subject}_trials_table.pqt'), engine='pyarrow')
    try:
        glm_hmm = glm_hmm.loc[glm_hmm['session'] == eid]
    except:
        pass
    glm_hmm_states = np.vstack(list(glm_hmm['glm-hmm_4']))

    # filter out trials with no choice
    choice_filter = np.where(trials['choice'] != 0)
    trials = {key: trials[key][choice_filter] for key in trials.keys()}
    pLeft = pLeft[choice_filter]
    trial_numbers = trial_numbers[choice_filter]
    glm_hmm_states = glm_hmm_states[choice_filter]

    # filter out trials with no contrastxf
    contrast_filter = ~np.logical_or(trials['contrastLeft'] == 0, trials['contrastRight'] == 0)
    trials = {key: trials[key][contrast_filter] for key in trials.keys()}
    pLeft = pLeft[contrast_filter]
    trial_numbers = trial_numbers[contrast_filter]
    glm_hmm_states = glm_hmm_states[contrast_filter]

    assert(trials['stimOff_times'].shape[0] == trials['feedback_times'].shape[0])
    assert(trials['stimOff_times'].shape[0] == trials['firstMovement_times'].shape[0])
    assert(trials['stimOff_times'].shape[0] == trials['contrastLeft'].shape[0])
    assert(trials['stimOff_times'].shape[0] == trials['contrastRight'].shape[0])
    assert(trials['stimOff_times'].shape[0] == trials['goCue_times'].shape[0])
    assert(trials['stimOff_times'].shape[0] == trials['choice'].shape[0])
    assert(trials['stimOff_times'].shape[0] == trials['response_times'].shape[0])
    assert(trials['stimOff_times'].shape[0] == trials['feedbackType'].shape[0])
    assert(trials['stimOff_times'].shape[0] == pLeft.shape[0])

    nan_idx = np.c_[np.isnan(trials['stimOn_times']), np.isnan(trials['firstMovement_times']), np.isnan(trials['goCue_times']),
                    np.isnan(trials['response_times']), np.isnan(trials['feedback_times']), np.isnan(trials['stimOff_times']),
                    np.isnan(pLeft)]
    kept_idx = np.sum(nan_idx, axis=1) == 0

    trials = {key: trials[key][kept_idx] for key in trials.keys()}
    pLeft = pLeft[kept_idx]
    glm_hmm_states = glm_hmm_states[kept_idx]
    trial_numbers = trial_numbers[kept_idx]

    # select trials
    if align_event == 'movement_onset':
        ref_event = trials['firstMovement_times']
        diff1 = ref_event - trials['stimOn_times']
        diff2 = trials['feedback_times'] - ref_event
        t_select1 = np.logical_and(diff1 > 0.0, diff1 < t_before - 0.1)
        t_select2 = np.logical_and(diff2 > 0.0, diff2 < t_after - 0.1)
        t_select = np.logical_and(t_select1, t_select2)

    trials = {key: trials[key][t_select] for key in trials.keys()}
    pLeft = pLeft[t_select]
    pLeft_last = np.roll(pLeft, 1)
    glm_hmm_states = glm_hmm_states[t_select]
    trial_numbers = trial_numbers[t_select]
    ref_event = ref_event[t_select]
        
    n_active_trials = ref_event.shape[0]
    
    # n_trials
    n_trials = n_active_trials
    print('number of trials found: {} (active: {})'.format(n_trials,n_active_trials))

    # Load in dlc
    left_dlc = one.load_object(eid, 'leftCamera', attribute=['dlc', 'features', 'times', 'ROIMotionEnergy'],
                               collection='alf')
    if left_dlc['times'].shape[0] != left_dlc['dlc'].shape[0]:
        left_offset = 0 #mtnn_eids[eid]
        left_dlc['times'] = left_dlc['times'][abs(left_offset):abs(left_offset) + left_dlc.shape[0]]
    assert (left_dlc['times'].shape[0] == left_dlc['dlc'].shape[0])

    left_dlc['dlc'] = dlc.likelihood_threshold(left_dlc['dlc'], threshold=0)

#     lpks = one.load_object(eid, 'leftCamera', attribute=['lightningPose'])['lightningPose']
    lpks = pd.read_parquet(save_data_path(figure='fig_mtnn').joinpath('lpks',
                                                                   f'{eid}._ibl_leftCamera.pose.pqt'), engine='pyarrow')
    left_dlc['dlc']['paw_l_x'] = lpks['paw_l_x']
    left_dlc['dlc']['paw_l_y'] = lpks['paw_l_y']
    left_dlc['dlc']['paw_r_x'] = lpks['paw_r_x']
    left_dlc['dlc']['paw_r_y'] = lpks['paw_r_y']
    
    left_dlc['dlc']['pupil_top_r_x'] = lpks['pupil_top_r_x']
    left_dlc['dlc']['pupil_top_r_y'] = lpks['pupil_top_r_y']
    left_dlc['dlc']['pupil_bottom_r_x'] = lpks['pupil_bottom_r_x']
    left_dlc['dlc']['pupil_bottom_r_y'] = lpks['pupil_bottom_r_y']
    left_dlc['dlc']['pupil_left_r_x'] = lpks['pupil_left_r_x']
    left_dlc['dlc']['pupil_left_r_y'] = lpks['pupil_left_r_y']
    left_dlc['dlc']['pupil_right_r_x'] = lpks['pupil_right_r_x']
    left_dlc['dlc']['pupil_right_r_y'] = lpks['pupil_right_r_y']
    
    
    # get licks
    # TO DO check if lick times ever nan
    try:
        lick_times = one.load_object(eid, 'licks', collection='alf')['times']
        if np.sum(np.isnan(lick_times)) > 0:
            lick_times = dlc.get_licks(left_dlc['dlc'], left_dlc['times'])
    except ALFObjectNotFound:
        lick_times = dlc.get_licks(left_dlc['dlc'], left_dlc['times'])

    # get right paw speed (closer to camera)
    paw_speed = dlc.get_speed(left_dlc['dlc'], left_dlc['times'], camera='left', feature='paw_r')

    # get nose speed
    nose_speed = dlc.get_speed(left_dlc['dlc'], left_dlc['times'], camera='left', feature='nose_tip')

    # get pupil diameter
    # TODO check in pupil diameter ever nan
#     if 'features' in left_dlc.keys():
#         pupil_diameter = left_dlc.pop('features')['pupilDiameter_smooth']
#         if np.sum(np.isnan(pupil_diameter)) > 0:
#             pupil_diameter = dlc.get_smooth_pupil_diameter(dlc.get_pupil_diameter(left_dlc['dlc']), 'left')
#     else:
#         pupil_diameter = dlc.get_smooth_pupil_diameter(dlc.get_pupil_diameter(left_dlc['dlc']), 'left')
    pupil_diameter = dlc.get_pupil_diameter(left_dlc['dlc'])

    # get wheel velocity
    wheel = one.load_object(eid, 'wheel')
    vel = wh.velocity(wheel['timestamps'], wheel['position'])
    wheel_timestamps = wheel['timestamps'][~np.isnan(vel)]
    vel = vel[~np.isnan(vel)]

    # Find responsive neurons
    # Baseline firing rate
    intervals = np.c_[trials['stimOn_times'] - 0.2, trials['stimOn_times']]
    counts, cluster_ids = get_spike_counts_in_bins(spikes['times'][spike_idx], spikes['clusters'][spike_idx], intervals)
    fr_base = counts / (intervals[:, 1] - intervals[:, 0])

    # Post-move firing rate
    intervals = np.c_[trials['firstMovement_times'] - 0.05, trials['firstMovement_times'] + 0.2]
    counts, cluster_ids = get_spike_counts_in_bins(spikes['times'][spike_idx], spikes['clusters'][spike_idx], intervals)
    fr_post_move = counts / (intervals[:, 1] - intervals[:, 0])

    sig_units, _, _ = compute_comparison_statistics(fr_base, fr_post_move, test='signrank')
    cluster_idx = cluster_idx[sig_units]
    good_clusters = cluster_id[sig_units]
    # Find the index of spikes that belong to the good clusters
    spike_idx = np.isin(spikes['clusters'], cluster_idx)

    print("(responsive) repeated site brain region counts: ", Counter(clusters['rep_site_acronym'][cluster_idx]))
    print("(responsive) number of good clusters: ", cluster_idx.shape[0])

    n_clusters = cluster_idx.shape[0]
    n_tbins = int((t_after + t_before) / bin_size)
    feature = np.zeros((n_clusters, n_trials, n_tbins, noise_offset + 1))
    print("feature tensor size: {}".format(feature.shape))

    # Create expected output array
    output, _ = bin_spikes2D(spikes['times'][spike_idx], spikes['clusters'][spike_idx], cluster_idx, ref_event,
                             t_before, t_after, bin_size)
    output = output / bin_size
    output = np.swapaxes(output, 0, 1)

    # Create feature array
    # session specific
    feature[:, :, :, session_offset + session_id] = 1

    # lab specific
    feature[:, :, :, lab_offset + lab_id] = 1

    # cluster specific
    for j, clust in notebook.tqdm(enumerate(cluster_idx)):

        # acronym
        acronym = clusters['rep_site_acronym'][clust]
        acronym_idx = acronym_dict[acronym]

        feature[j, :, :, 0] = j
        
        feature[j, :, :, xyz_offset] = clusters['x'][clust]
        feature[j, :, :, xyz_offset+1] = clusters['y'][clust]
        feature[j, :, :, xyz_offset+2] = clusters['z'][clust]
        
        feature[j, :, :, acronym_offset+acronym_idx] = 1
        
        # spike max ptp
        feature[j, :, :, max_ptp_offset] = clusters['amps'][clust]
        
        # spike wf width
        feature[j, :, :, wf_width_offset] = clusters['peakToTrough'][clust]

    # trial specific
    # wheel velocity
    bin_vel, _ = bin_norm(wheel_timestamps, ref_event, t_before, t_after, bin_size, weights=vel)
    # left motion energy
    bin_left_me, _ = bin_norm(left_dlc['times'], ref_event, t_before, t_after, bin_size, weights=left_dlc['ROIMotionEnergy'])
    # paw speed
    bin_paw_speed, _ = bin_norm(left_dlc['times'], ref_event, t_before, t_after, bin_size, weights=paw_speed)
    # nose speed
    bin_nose_speed, _ = bin_norm(left_dlc['times'], ref_event, t_before, t_after, bin_size, weights=nose_speed)
    # pupil diameter
    bin_pup_dia, _ = bin_norm(left_dlc['times'], ref_event, t_before, t_after, bin_size, weights=pupil_diameter)
    # lick times
    bin_lick, _ = bin_norm(lick_times, ref_event, t_before, t_after, bin_size, weights=np.ones_like(lick_times))
    # goCue times
    bin_go_cue, _ = bin_spikes(trials['goCue_times'], ref_event, t_before, t_after, bin_size)
    #firt movement times
    bin_first_move, _ = bin_spikes(trials['firstMovement_times'], ref_event, t_before, t_after, bin_size)
    # choice
    bin_choice, _ = bin_spikes(trials['response_times'], ref_event, t_before, t_after, bin_size)
    # reward
    bin_reward, _ = bin_spikes(trials['feedback_times'], ref_event, t_before, t_after, bin_size)
    # stimulus contrast
    bin_stim_on, _ = bin_spikes(trials['stimOn_times'], ref_event, t_before, t_after, bin_size)

    for k in notebook.tqdm(range(n_active_trials)):

        # dlc
        feature[:, k, :, paw_offset] = bin_paw_speed[k, :]
        feature[:, k, :, nose_offset] = bin_nose_speed[k, :]
        feature[:, k, :, pupil_offset] = bin_pup_dia[k, :]
        feature[:, k, :, left_me_offset] = bin_left_me[k, :]

        # goCue
        feature[:, k, :, goCue_offset] = bin_go_cue[k, :]
                
        # firstMovement
        feature[:, k, :, firstMovement_offset] = bin_first_move[k, :]

        # choice
        if trials['choice'][k] == -1:
            feature[:, k, :, choice_offset] = bin_choice[k, :]
        elif trials['choice'][k] == 1:
            feature[:, k, :, choice_offset+1] = bin_choice[k, :]

        # reward
        if trials['feedbackType'][k] == -1:
            feature[:, k, :, reward_offset] = bin_reward[k, :]
        else:
            feature[:, k, :, reward_offset+1] = bin_reward[k, :]

        # wheel velocity
        feature[:, k, :, wheel_offset] = bin_vel[k, :]
            
        # lick
        feature[:, k, :, lick_offset] = bin_lick[k, :]
            
        # mouse prior
        feature[:, k, :, pLeft_offset] = pLeft[k]
        feature[:, k, :, pLeft_last_offset] = pLeft_last[k]
        
        # decision strategy
        feature[:, k, :, glmhmm_offset:acronym_offset] = glm_hmm_states[k]
        
        # stimulus
        if trials['contrastLeft'][k] != 0 and not np.isnan(trials['contrastLeft'][k]):
            feature[:, k, :, stimulus_offset] = trials['contrastLeft'][k] * bin_stim_on[k, :]
        elif trials['contrastRight'][k] != 0 and not np.isnan(trials['contrastRight'][k]):
            feature[:, k, :, stimulus_offset+1] = trials['contrastRight'][k] * bin_stim_on[k, :]

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
        feature_list.append(np.load(save_data_path(figure='fig_mtnn').joinpath('original_data', f'{eid}_feature.npy')))
        output_list.append(np.load(save_data_path(figure='fig_mtnn').joinpath('original_data', f'{eid}_output.npy')))
        cluster_number_list.append(np.load(save_data_path(figure='fig_mtnn').joinpath('original_data', f'{eid}_clusters.npy')))
        session_list.append(np.load(save_data_path(figure='fig_mtnn').joinpath('original_data', f'{eid}_session_info.npy'), allow_pickle=True))
        trial_number_list.append(np.load(save_data_path(figure='fig_mtnn').joinpath('original_data', f'{eid}_trials.npy')))
        
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


def select_high_fr_neurons(feature, output, clusters,
                           neuron_id_start=0, threshold=0.0, max_n_neurons=15):
    select = np.mean(output, axis=(1, 2)) >= threshold
    feature_subset = feature[select]
    if feature_subset.shape[0] > max_n_neurons:
        select2 = rng.choice(np.arange(feature_subset.shape[0]), size=max_n_neurons, replace=False)
    else:
        select2 = np.arange(feature_subset.shape[0])
    feature_subset = feature_subset[select2]
    for i in range(feature_subset.shape[0]):
        feature_subset[i, :, :, 0] = neuron_id_start + i

    return feature_subset, output[select][select2], clusters[select][select2]


def preprocess_feature(feature_concat,
                       xyz_stat=None,
                       dlc_stat=None,
                       wheel_stat=None,
                       max_ptp_stat=None,
                       wf_width_stat=None):
    # normalize truncated_feature: dlc features + xyz location
    # first normalize xyz
    if xyz_stat is None:
        x_max = feature_concat[:, :, xyz_offset].max()
        x_min = feature_concat[:, :, xyz_offset].min()
        y_max = feature_concat[:, :, xyz_offset + 1].max()
        y_min = feature_concat[:, :, xyz_offset + 1].min()
        z_max = feature_concat[:, :, xyz_offset + 2].max()
        z_min = feature_concat[:, :, xyz_offset + 2].min()
        xyz_stat = (x_max, x_min, y_max, y_min, z_max, z_min)
    else:
        x_max, x_min, y_max, y_min, z_max, z_min = xyz_stat

    feature_concat[:, :, xyz_offset] = 0.1 + 0.9 * (feature_concat[:, :, xyz_offset] - x_min) / (x_max - x_min)
    feature_concat[:, :, xyz_offset + 1] = 0.1 + 0.9 * (feature_concat[:, :, xyz_offset + 1] - y_min) / (y_max - y_min)
    feature_concat[:, :, xyz_offset + 2] = 0.1 + 0.9 * (feature_concat[:, :, xyz_offset + 2] - z_min) / (z_max - z_min)

    if dlc_stat is None:
        tmp_dlc_stat = []
    # next normalize dlc features
    for i in range(stimulus_offset - paw_offset):
        idx = paw_offset + i

        if dlc_stat is None:
            feature_min = feature_concat[:, :, idx].min()
            feature_max = feature_concat[:, :, idx].max()
            tmp_dlc_stat.append((feature_min, feature_max))
        else:
            feature_min, feature_max = dlc_stat[i]

        feature_concat[:, :, idx] = 0.1 + 0.9 * (feature_concat[:, :, idx] - feature_min) / (feature_max - feature_min)
    if dlc_stat is None:
        dlc_stat = tmp_dlc_stat

    # next normalize wheel
    if wheel_stat is None:
        wheel_min = feature_concat[:, :, wheel_offset].min()
        wheel_max = feature_concat[:, :, wheel_offset].max()
        wheel_stat = (wheel_min, wheel_max)
    else:
        wheel_min, wheel_max = wheel_stat

    feature_concat[:, :, wheel_offset] = -1 + 2 * (feature_concat[:, :, wheel_offset] - wheel_min) / (wheel_max - wheel_min)

    # next normalize max_ptp
    if max_ptp_stat is None:
        max_ptp_min = feature_concat[:, :, max_ptp_offset].min()
        max_ptp_max = feature_concat[:, :, max_ptp_offset].max()
        max_ptp_stat = (max_ptp_min, max_ptp_max)
    else:
        max_ptp_min, max_ptp_max = max_ptp_stat

    feature_concat[:, :, max_ptp_offset] = 0.1 + 0.9 * (feature_concat[:, :, max_ptp_offset] - max_ptp_min) / (
                max_ptp_max - max_ptp_min)

    # next normalize wf_width
    if wf_width_stat is None:
        wf_width_min = feature_concat[:, :, wf_width_offset].min()
        wf_width_max = feature_concat[:, :, wf_width_offset].max()
        wf_width_stat = (wf_width_min, wf_width_max)
    else:
        wf_width_min, wf_width_max = wf_width_stat

    feature_concat[:, :, wf_width_offset] = 0.1 + 0.9 * (feature_concat[:, :, wf_width_offset] - wf_width_min) / (
                wf_width_max - wf_width_min)

    # noise
    noise = rng.normal(loc=0.0, scale=1.0, size=feature_concat.shape[:-1])
    feature_concat[:, :, noise_offset] = noise

    return feature_concat, xyz_stat, dlc_stat, wheel_stat, max_ptp_stat, wf_width_stat

def load_trials_df(eid, one=None, maxlen=None, t_before=0., t_after=0., ret_wheel=False,
                   ret_abswheel=False, wheel_binsize=0.02, addtl_types=[], 
                   align_event='stimOn_times', keeptrials=None):
    """
    Generate a pandas dataframe of per-trial timing information about a given session.
    Each row in the frame will correspond to a single trial, with timing values indicating timing
    session-wide (i.e. time in seconds since session start). Can optionally return a resampled
    wheel velocity trace of either the signed or absolute wheel velocity.

    The resulting dataframe will have a new set of columns, trial_start and trial_end, which define
    via t_before and t_after the span of time assigned to a given trial.
    (useful for bb.modeling.glm)

    Parameters
    ----------
    eid : [str, UUID, Path, dict]
        Experiment session identifier; may be a UUID, URL, experiment reference string
        details dict or Path
    one : oneibl.one.OneAlyx, optional
        one object to use for loading. Will generate internal one if not used, by default None
    maxlen : float, optional
        Maximum trial length for inclusion in df. Trials where feedback - response is longer
        than this value will not be included in the dataframe, by default None
    t_before : float, optional
        Time before stimulus onset to include for a given trial, as defined by the trial_start
        column of the dataframe. If zero, trial_start will be identical to stimOn, by default 0.
    t_after : float, optional
        Time after feedback to include in the trail, as defined by the trial_end
        column of the dataframe. If zero, trial_end will be identical to feedback, by default 0.
    ret_wheel : bool, optional
        Whether to return the time-resampled wheel velocity trace, by default False
    ret_abswheel : bool, optional
        Whether to return the time-resampled absolute wheel velocity trace, by default False
    wheel_binsize : float, optional
        Time bins to resample wheel velocity to, by default 0.02
    addtl_types : list, optional
        List of additional types from an ONE trials object to include in the dataframe. Must be
        valid keys to the dict produced by one.load_object(eid, 'trials'), by default empty.

    Returns
    -------
    pandas.DataFrame
        Dataframe with trial-wise information. Indices are the actual trial order in the original
        data, preserved even if some trials do not meet the maxlen criterion. As a result will not
        have a monotonic index. Has special columns trial_start and trial_end which define start
        and end times via t_before and t_after
    """
    if not one:
        one = ONE()

    if ret_wheel and ret_abswheel:
        raise ValueError('ret_wheel and ret_abswheel cannot both be true.')

    # Define which datatypes we want to pull out
    trialstypes = ['choice',
                   'probabilityLeft',
                   'feedbackType',
                   'feedback_times',
                   'contrastLeft',
                   'contrastRight',
                   'goCue_times',
                   'stimOn_times']
    trialstypes.extend(addtl_types)

    # A quick function to remap probabilities in those sessions where it was not computed correctly
    def remap_trialp(probs):
        # Block probabilities in trial data aren't accurate and need to be remapped
        validvals = np.array([0.2, 0.5, 0.8])
        diffs = np.abs(np.array([x - validvals for x in probs]))
        maps = diffs.argmin(axis=1)
        return validvals[maps]

    trials = one.load_object(eid, 'trials', collection='alf')
    starttimes = trials.stimOn_times
    endtimes = trials.feedback_times
    tmp = {key: value for key, value in trials.items() if key in trialstypes}

    if keeptrials is None:
        if maxlen is not None:
            with np.errstate(invalid='ignore'):
                keeptrials = (endtimes - starttimes) <= maxlen
        else:
            keeptrials = range(len(starttimes))
    trialdata = {x: tmp[x][keeptrials] for x in trialstypes}
    trialdata['probabilityLeft'] = remap_trialp(trialdata['probabilityLeft'])
    trialsdf = pd.DataFrame(trialdata)
    if maxlen is not None:
        trialsdf.set_index(np.nonzero(keeptrials)[0], inplace=True)
    trialsdf['trial_start'] = trialsdf[align_event] - t_before
    trialsdf['trial_end'] = trialsdf[align_event] + t_after
    tdiffs = trialsdf['trial_end'] - np.roll(trialsdf['trial_start'], -1)
    if np.any(tdiffs[:-1] > 0):
        logging.warning(f'{sum(tdiffs[:-1] > 0)} trials overlapping due to t_before and t_after '
                        'values. Try reducing one or both!')
    if not ret_wheel and not ret_abswheel:
        return trialsdf

    wheel = one.load_object(eid, 'wheel', collection='alf')
    whlpos, whlt = wheel.position, wheel.timestamps
    starttimes = trialsdf['trial_start']
    endtimes = trialsdf['trial_end']
    wh_endlast = 0
    trials = []
    for (start, end) in np.vstack((starttimes, endtimes)).T:
        wh_startind = np.searchsorted(whlt[wh_endlast:], start) + wh_endlast
        wh_endind = np.searchsorted(whlt[wh_endlast:], end, side='right') + wh_endlast + 4
        wh_endlast = wh_endind
        tr_whlpos = whlpos[wh_startind - 1:wh_endind + 1]
        tr_whlt = whlt[wh_startind - 1:wh_endind + 1] - start
        tr_whlt[0] = 0.  # Manual previous-value interpolation
        whlseries = TimeSeries(tr_whlt, tr_whlpos, columns=['whlpos'])
        whlsync = sync(wheel_binsize, timeseries=whlseries, interp='previous')
        trialstartind = np.searchsorted(whlsync.times, 0)
        trialendind = np.ceil((end - start) / wheel_binsize).astype(int)
        trpos = whlsync.values[trialstartind:trialendind + trialstartind]
        whlvel = trpos[1:] - trpos[:-1]
        whlvel = np.insert(whlvel, 0, 0)
        if np.abs((trialendind - len(whlvel))) > 0:
            raise IndexError('Mismatch between expected length of wheel data and actual.')
        if ret_wheel:
            trials.append(whlvel)
        elif ret_abswheel:
            trials.append(np.abs(whlvel))
    trialsdf['wheel_velocity'] = trials
    return trialsdf
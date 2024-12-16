import numpy as np
from tqdm import tqdm

from brainbox.io.one import SpikeSortingLoader, SessionLoader
import brainbox.behavior.wheel as wh

"""
IBL BWM data process
"""
def load_spikes_from_pid(one, ba, pid):
    """
    - iterate over pids of this eid,
        - read spike trains, cluster information 
    """
    # Manually combine spike data from all pids
    spikes_eid = None
    clusters_eid = None
    _nclusters = 0
    # load all pids from this session
    ssl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    _spikes, _clusters, _channels = ssl.load_spike_sorting(dataset_types=['clusters.amps', 'clusters.peakToTrough'],
                                                           revision='2024-03-22', enforce_version=False)
    _clusters = ssl.merge_clusters(_spikes, _clusters, _channels)
    if _clusters is None:
        print(f"!!!! no spiking data in pid {pid} !!!")

    spikes_eid = _append_spike_data(spikes_eid, _spikes, _nclusters)
    clusters_eid = _append_clusters_data(ba.regions, clusters_eid, _clusters, _nclusters)

    _nclusters += np.max((len(_clusters['atlas_id']), _clusters['atlas_id'].max()))
    return {"spikes": spikes_eid,
            "clusters": clusters_eid,
            "nclusters": _nclusters}


def _append_spike_data(spikes_eid, spikes, _nclusters):
    if spikes_eid is None:
        spikes_eid = {'times': [], 'clusters': []}
    spikes_eid['times'] = np.concatenate((spikes_eid['times'], spikes['times']), 0)
    spikes_eid['clusters'] = np.concatenate((spikes_eid['clusters'], spikes['clusters'] + _nclusters), 0)
    return spikes_eid


def _append_clusters_data(br, clusters_eid, clusters, _nclusters):
    if clusters_eid is None:
        clusters_eid = {'depths': [], 'cluster_id': [], 'firing_rate': [], 'acronym': [], 'Cosmos': [], 'atlas_id': [],
                        'label': [], 'uuids': []}
    clusters_eid['depths'] = np.concatenate((clusters_eid['depths'], clusters['depths']), 0)
    clusters_eid['label'] = np.concatenate((clusters_eid['label'], clusters['label']), 0)
    clusters_eid['uuids'] = np.concatenate((clusters_eid['uuids'], clusters['uuids']), 0)
    clusters_eid['cluster_id'] = np.concatenate((clusters_eid['cluster_id'], clusters['cluster_id'] + _nclusters), 0)
    clusters_eid['firing_rate'] = np.concatenate((clusters_eid['firing_rate'], clusters['firing_rate']), 0)
    clusters_eid['acronym'] = np.concatenate(
        (clusters_eid['acronym'], br.acronym2acronym(clusters['acronym'], mapping='Beryl')), 0)  # using Beryl mapping
    clusters_eid['Cosmos'] = np.concatenate(
        (clusters_eid['Cosmos'], br.acronym2acronym(clusters['acronym'], mapping='Cosmos')), 0)  # using Cosmos mapping
    clusters_eid['atlas_id'] = np.concatenate((clusters_eid['atlas_id'], clusters['atlas_id']), 0)
    return clusters_eid


def _get_spike_train_tid(clusters_g, spikes_g):
    c_idxs = clusters_g['cluster_id']
    c_idx2ci = {};
    ci2c_idx = {}
    for ci, c_idx in enumerate(c_idxs):
        ci2c_idx[ci] = c_idx
        c_idx2ci[c_idx] = ci
    spike_cis = np.array([c_idx2ci[c_idx] for c_idx in spikes_g['clusters']])
    spike_train = np.c_[spikes_g['times'], spike_cis]
    return spike_train


def get_spikecountmatrix(trials, spike_train, N,
                         spsdt, Twindow, t_bf_stimOn=0.4):
    num_trial = len(trials);  # N = np.max(spike_train[:,1]).astype(int)+1
    spsT = int(Twindow / spsdt) + 1;
    spst_bins = np.arange(spsT + 1) * spsdt

    spike_count_matrix = []
    for ti in tqdm(range(num_trial)):
        trial = trials.iloc[ti]
        (trial_st, trial_et) = (trial['stimOn_times'] - t_bf_stimOn, trial['stimOn_times'] - t_bf_stimOn + Twindow)
        ## spike
        mask = np.logical_and(
            spike_train[:, 0] >= trial_st,
            spike_train[:, 0] < trial_et
        )
        sub_spike_train = spike_train[mask]
        spike_count = np.zeros([spsT, N])  # initialize
        if sub_spike_train.shape[0] == 0:
            # no spike
            spike_count_matrix.append(spike_count)
        else:
            ### compute spike count matrix
            sub_spike_train[:, 0] = sub_spike_train[:, 0] - trial_st
            units = sub_spike_train[:, 1].astype(int)
            t_bins = np.digitize(sub_spike_train[:, 0], spst_bins, right=False) - 1
            np.add.at(spike_count, (t_bins, units), 1)
            spike_count_matrix.append(spike_count / spsdt)
    spike_count_matrix = np.array(spike_count_matrix)
    return spike_count_matrix


def get_behavior(trials):
    num_trial = len(trials)

    behavior = [];
    for ti in tqdm(range(num_trial)):
        trial = trials.iloc[ti]

        ## behavior
        lorr = 1. if ~np.isnan(trial['contrastLeft']) else -1.  # 1. for left and -1. for right
        contrast = np.nan_to_num(trial['contrastLeft']) + np.nan_to_num(trial['contrastRight'])
        _beh = f"{trial['probabilityLeft']}_{lorr}_{contrast}_{trial['choice']}"
        _beh += f"_{trial['choice_last']}_{trial['reward_last']}"
        behavior.append(_beh)

    behavior = np.array(behavior)
    return behavior


def get_timeline(trials,
                 spsdt, Twindow, t_bf_stimOn=0.4):
    num_trial = len(trials)

    timeline = []
    for ti in tqdm(range(num_trial)):
        trial = trials.iloc[ti]
        (trial_st, trial_et) = (trial['stimOn_times'] - t_bf_stimOn, trial['stimOn_times'] - t_bf_stimOn + Twindow)

        ## timeline
        _toadd = []
        for tp in [trial['stimOn_times'], trial['firstMovement_times'], trial['response_times'],
                   trial['stimOff_times']]:
            if np.isnan(tp) or tp >= trial_et:
                _toadd.append(None)
            elif tp < trial_st:
                _toadd.append(0)
            else:
                _toadd.append(int((tp - trial_st) / spsdt))
        timeline.append(_toadd)

    timeline = np.array(timeline)
    return timeline


def get_wheelvelocity(trials, wheel_t, wheel_pos, Fs,
                      spsdt, Twindow, t_bf_stimOn=0.4):
    num_trial = len(trials)
    spsT = int(Twindow / spsdt) + 1

    wheel_vel = []
    for ti in tqdm(range(num_trial)):
        trial = trials.iloc[ti]
        (trial_st, trial_et) = (trial['stimOn_times'] - t_bf_stimOn, trial['stimOn_times'] - t_bf_stimOn + Twindow)

        ## wheel velocity
        traces = wh.traces_by_trial(wheel_t, wheel_pos, start=[trial_st], end=[trial_et + spsdt])
        vel, _ = wh.velocity_filtered(traces[0][1], Fs)
        vel = np.concatenate((vel, np.zeros(spsT - vel.shape[0])))
        wheel_vel.append(vel)

    wheel_vel = np.array(wheel_vel)[:, :, np.newaxis]  # (K,T,1)
    return wheel_vel


def get_interpbeh(trials, beh_rawdata, beh_k, cameras,
                  spsdt, Twindow, t_bf_stimOn=0.4):
    num_trial = len(trials)

    beh = []
    for ti in tqdm(range(num_trial)):
        trial = trials.iloc[ti]
        (trial_st, trial_et) = (trial['stimOn_times'] - t_bf_stimOn, trial['stimOn_times'] - t_bf_stimOn + Twindow)

        yinterps = []
        for cam in cameras:
            if cam is None:
                mask = np.logical_and(
                    beh_rawdata["times"] >= trial_st,
                    beh_rawdata["times"] < trial_et
                )
                xvals = np.arange(trial_st, trial_et + 0.001, spsdt)
                yinterp = np.zeros_like(xvals)
                if np.sum(mask) > 0:
                    x = beh_rawdata["times"][mask]
                    y = beh_rawdata[beh_k][mask]
                    yinterp = np.interp(xvals, x, y)
            else:
                cam = cam + "Camera"
                mask = np.logical_and(
                    beh_rawdata[cam]["times"] >= trial_st,
                    beh_rawdata[cam]["times"] < trial_et
                )
                xvals = np.arange(trial_st, trial_et + 0.001, spsdt)
                yinterp = np.zeros_like(xvals)
                if np.sum(mask) > 0:
                    x = beh_rawdata[cam]["times"][mask]
                    y = beh_rawdata[cam][beh_k][mask]
                    yinterp = np.interp(xvals, x, y)
            yinterps.append(yinterp)
        beh.append(np.asarray(yinterps).T)

    beh = np.array(beh)  # (K, T, 1/2)
    return beh


def get_tongue(trials, licks_rawdata,
               spsdt, Twindow, t_bf_stimOn=0.4):
    num_trial = len(trials)
    spsT = int(Twindow / spsdt) + 1;
    spst_bins = np.arange(spsT + 1) * spsdt

    licks = []
    for ti in tqdm(range(num_trial)):
        trial = trials.iloc[ti]
        (trial_st, trial_et) = (trial['stimOn_times'] - t_bf_stimOn, trial['stimOn_times'] - t_bf_stimOn + Twindow)

        ## lick
        mask = np.logical_and(
            licks_rawdata["times"] >= trial_st,
            licks_rawdata["times"] < trial_et
        )
        times = licks_rawdata["times"][mask]
        lick_count = np.zeros([spsT, 1])  # initialize
        if times.shape[0] == 0:
            # no lick
            licks.append(lick_count)
        else:
            ### compute lick matrix
            times = times - trial_st
            units = np.zeros(len(times), dtype=int)
            t_bins = np.digitize(times, spst_bins, right=False) - 1
            np.add.at(lick_count, (t_bins, units), 1)
            licks.append(lick_count / spsdt)

    licks = np.array(licks)  # (K, T, 1)
    return licks


def filter_trials_func(trials,
                       remove_nan_event=(True, 'firstMovement_times', 'stimOn_times', 'response_times'),
                       remove_timeextreme_event=(True, 1.6),
                       remove_no_choice=True,
                       remove_wrong_choice=False,
                       remove_zerocontrast=False):
    index_toremove = []
    if remove_nan_event[0]:
        for k in remove_nan_event[1:]:
            nan_index = np.where(np.isnan(trials[k]))[0]
            index_toremove += [nan_index]
    if remove_timeextreme_event[0]:
        exttime = remove_timeextreme_event[1]
        invalid_index = np.where((trials['firstMovement_times'] - trials['stimOn_times'] >= exttime) | (
                    trials['firstMovement_times'] - trials['stimOn_times'] < 0.0))[0]
        invalid_index2 = np.where((trials['response_times'] - trials['stimOn_times'] >= exttime))[0]
        index_toremove += [invalid_index, invalid_index2]
    if remove_no_choice:
        nan_index4 = np.where(trials['choice'] == 0.)[0]
        index_toremove += [nan_index4]
    if remove_wrong_choice:
        wrong_index = np.where(trials.feedbackType == -1.)[0]
        index_toremove += [wrong_index]
    if remove_zerocontrast:
        zc_index2 = np.where((trials.contrastLeft == 0.) | (trials.contrastRight == 0.))[0]
        index_toremove += [zc_index2]
    index_toremove = np.concatenate(index_toremove, axis=0)
    print(f"{len(index_toremove)} trials being dropped out of {trials.shape[0]}")

    return trials.drop(index=index_toremove)


def filter_neurons_func(eid_spikes_res,
                        remove_frextreme=(True, 3., 50.),
                        only_goodneuron=(False),
                        only_area=(False)):
    """
    :eid_spikes_res: return from load_spikes_from_eid,
    : include values of spikes (indicate times and clusters), and clusters
    """

    def _good_neuron(clusters_eid):
        return (clusters_eid['firing_rate'] > remove_frextreme[1]) & (clusters_eid['firing_rate'] < remove_frextreme[2])

    clusters_eid = eid_spikes_res['clusters']
    spikes_eid = eid_spikes_res['spikes']
    toselect_cluster_mask = np.ones_like(clusters_eid['firing_rate']).astype(bool)
    if only_goodneuron:
        toselect_cluster_mask &= clusters_eid["label"] == 1
    if remove_frextreme:
        good_cluster_mask = _good_neuron(clusters_eid)
        toselect_cluster_mask &= good_cluster_mask
    if only_area:
        rightarea_cluster_mask = np.isin(clusters_eid['acronym'], only_area[1])
        toselect_cluster_mask &= rightarea_cluster_mask
    print(f"{np.sum(~toselect_cluster_mask)} neurons being dropped out of {len(toselect_cluster_mask)}")
    toselect_cluster_IDs = clusters_eid['cluster_id'][toselect_cluster_mask]
    # Filter the clusters accordingly:
    clusters_g = {key: val[toselect_cluster_mask] for key, val in clusters_eid.items()}
    # Filter the spikes accordingly:
    toselect_spk_indx = np.where(np.isin(spikes_eid['clusters'], toselect_cluster_IDs))
    spikes_g = {key: val[toselect_spk_indx] for key, val in spikes_eid.items()}
    return {"spikes_g": spikes_g,
            "clusters_g": clusters_g}


def load_data_from_pid(pid, one, ba,
                       filter_trials, filter_neurons,
                       min_trials=10, min_neurons=10, spsdt=5e-3, Twindow=2., t_bf_stimOn=0.4,
                       load_motion_energy=True, load_wheel_velocity=True, load_tongue=True):
    # ---------------------------------------------------

    eid, _ = one.pid2eid(pid)
    # Load trials data
    sl = SessionLoader(eid=eid, one=one)
    try:
        sl.load_session_data() # load trials, wheel, whisker ME, pose, pupil
        if sl.data_info.loc[0, 'is_loaded'] == False: # trials
            print(f"No trials in session={eid}")
            return None
    except:
        return None

    sl.trials["choice_last"] = np.concatenate([[0.], sl.trials.choice[:-1].values])  # 1., -1., 0.
    sl.trials["reward_last"] = np.concatenate([[-1.], sl.trials.feedbackType[:-1].values])  # 1., -1.

    sl.trials = filter_trials(sl.trials)
    # N trial count
    num_trial = len(sl.trials)
    if num_trial < min_trials:
        print(f"Not enough ({num_trial}) trials in session={eid}")
        return None

    # ---------------------------------------------------
    # Load whisker ME data
    if load_motion_energy:
        if ('leftCamera' not in sl.motion_energy) or ('rightCamera' not in sl.motion_energy):
            print(f"No whisker motion energy in session={eid}")
            return None
        if ('whiskerMotionEnergy' not in sl.motion_energy['leftCamera']) or ('whiskerMotionEnergy' not in sl.motion_energy['rightCamera']):
            print(f"No motion energy in session={eid}")
            return None
    
    # ---------------------------------------------------
    # Load wheel data
    if load_wheel_velocity:
        if sl.data_info.loc[1, 'is_loaded'] == False: # wheel
            print(f"No wheel velocity in session={eid}")
            return None

    # ---------------------------------------------------
    # Load tongue data
    if load_tongue:
        try:
            licks_rawdata = one.load_object(eid, 'licks', collection='alf')
        except:
            print(f"No tongue in session={eid}")
            return None

    # ---------------------------------------------------
    # Load spike data
    res = load_spikes_from_pid(one, ba, pid)

    # ---------------------------------------------------
    # Find cluster index with reasonable firing rates
    res_g = filter_neurons(res)
    # ---------------------------------------------------
    # N neuronal units in total
    num_neuron = len(res_g["clusters_g"]["cluster_id"])
    if num_neuron < min_neurons:
        print(f"Not enough ({num_neuron}) neurons in session={eid}")
        return None

    ### make spike count matrix and decision matrix
    trial_setup = {"spsdt": spsdt, "Twindow": Twindow, "t_bf_stimOn": t_bf_stimOn}
    spike_train_tid = _get_spike_train_tid(res_g["clusters_g"], res_g["spikes_g"])
    if spike_train_tid.shape[0] == 0:
        return None
    spike_count_matrix = get_spikecountmatrix(sl.trials, spike_train_tid, len(res_g["clusters_g"]["cluster_id"]),
                                              **trial_setup)
    behavior = get_behavior(sl.trials)
    timeline = get_timeline(sl.trials, **trial_setup)

    if load_wheel_velocity:
        wheel_vel = get_interpbeh(sl.trials, sl.wheel, "velocity", [None],
                                   **trial_setup)
    else:
        wheel_vel = None

    if load_motion_energy:
        whisker_motion = get_interpbeh(sl.trials, sl.motion_energy, "whiskerMotionEnergy", ['left', 'right'],
                                       **trial_setup)
    else:
        whisker_motion = None
    
    if load_tongue:
        licks = get_tongue(sl.trials, licks_rawdata, **trial_setup)
    else:
        licks = None

    return {"spike_count_matrix": spike_count_matrix,
            "clusters_g": res_g["clusters_g"],  # contain information as in _append_clusters_data()
            "behavior": behavior,
            "timeline": timeline,
            "wheel_vel": wheel_vel,
            "whisker_motion": whisker_motion,
            "licks": licks,
            }

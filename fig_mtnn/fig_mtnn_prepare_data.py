from one.api import ONE
import numpy as np
import pandas as pd
from tqdm import notebook
from numpy.random import normal
from sklearn.linear_model import RidgeCV

from iblatlas.atlas import AllenAtlas
import fig_mtnn.modeling.linear as lm
import fig_mtnn.modeling.utils as mut
import brainbox.io.one as bbone

from reproducible_ephys_functions import save_data_path, save_dataset_info, repo_path
from fig_mtnn.utils import *
from fig_mtnn.glm import generate_design, generate_design_full_mtnn_cov, bases, bases_full_mtnn_cov, binwidth, t_before, t_after, GLMPredictor
from fig_mtnn.simulate import simulate_cell, concat_simcell_data, to_mtnn_form
from fig_mtnn.fig_mtnn_load_data import download_priors, download_glm_hmm, download_lp

import matplotlib.pyplot as plt

from collections import defaultdict

data_path = save_data_path(figure='fig_mtnn')

# rng = np.random.default_rng(seed=0b01101001 + 0b01100010 + 0b01101100)
# rng = np.random.default_rng(seed=10234567)
rng = np.random.default_rng(seed=6671015)
alphas = (0.1, 0.5, 1.0, 3.0, 5.0, 7.0, 10.0)

def prepare_data(one):
    brain_atlas = AllenAtlas()
    eids = get_mtnn_eids()
    insertions = get_traj(eids)
    prepare_mtnn_data(eids, insertions, one, brain_atlas=brain_atlas)
    prepare_glm_and_simulated_data(insertions, one, brain_atlas=brain_atlas)
    prepare_glm_data_full_mtnn_cov(insertions, one, brain_atlas=brain_atlas)

def prepare_mtnn_data(eids, insertions, one, brain_atlas=None):

    download_priors()
    download_glm_hmm()
    download_lp()

    feature_list = []
    output_list = []
    cluster_number_list = []
    trial_number_list = []
    session_list = []
    session_count = {('hoferlab', 'mrsicflogellab'): 0,
                     ('mainenlab',): 0, ('churchlandlab',): 0,
                     ('cortexlab',): 0, ('danlab',): 0,
                     ('angelakilab',): 0, ('churchlandlab_ucla',): 0, ('steinmetzlab',): 0}

    for i, ins in enumerate(insertions):
        feature, output, cluster_numbers, trial_numbers = featurize(i, ins, one, session_count, brain_atlas=brain_atlas)
        feature_list.append(feature)
        output_list.append(output)
        cluster_number_list.append(cluster_numbers)
        session_list.append(ins)
        trial_number_list.append(trial_numbers)

    save_path = data_path.joinpath('original_data')
    save_path.mkdir(exist_ok=True, parents=True)

    for i in range(len(feature_list)):
        print(session_list[i]['session']['id'])
        print(feature_list[i].shape)
        np.save(save_path.joinpath(f'{session_list[i]["session"]["id"]}_feature.npy'), feature_list[i])

    for i in range(len(output_list)):
        print(session_list[i]['session']['id'])
        print(output_list[i].shape)
        np.save(save_path.joinpath(f'{session_list[i]["session"]["id"]}_output.npy'), output_list[i])

    for i in range(len(cluster_number_list)):
        print(session_list[i]['session']['id'])
        print(cluster_number_list[i].shape)
        np.save(save_path.joinpath(f'{session_list[i]["session"]["id"]}_clusters.npy'), cluster_number_list[i])

    for i in range(len(session_list)):
        print(session_list[i]['session']['id'])
        np.save(save_path.joinpath(f'{session_list[i]["session"]["id"]}_session_info.npy'), session_list[i])

    for i in range(len(trial_number_list)):
        print(session_list[i]['session']['id'])
        print(trial_number_list[i].shape)
        np.save(save_path.joinpath(f'{session_list[i]["session"]["id"]}_trials.npy'), trial_number_list[i])

    # feature_list, output_list, cluster_number_list, session_list, trial_number_list = load_original(eids)

    total_n_neurons = 0
    shape_list = []
    output_subset_list = []
    cluster_subset_list = []
    for i in notebook.tqdm(range(len(insertions))):
        feature_subset, output_subset, clusters_subset = select_high_fr_neurons(feature_list[i],
                                                                                output_list[i],
                                                                                cluster_number_list[i],
                                                                                neuron_id_start=total_n_neurons,
                                                                                threshold=5.0,
                                                                                max_n_neurons=500)#15)
        total_n_neurons += feature_subset.shape[0]
        print('{}/{} remaining'.format(feature_subset.shape[0], feature_list[i].shape[0]))
        print('{}/{} removed'.format(feature_list[i].shape[0] - feature_subset.shape[0], feature_list[i].shape[0]))
        shape_list.append(feature_subset.shape)
        output_subset_list.append(output_subset)
        cluster_subset_list.append(clusters_subset)

        if i == 0:
            feature_concat = feature_subset.reshape((-1,) + feature_subset.shape[-2:])
        else:
            feature_concat = np.concatenate((feature_concat, feature_subset.reshape((-1,) + feature_subset.shape[-2:])))
    print('feature_concat shape: {}'.format(feature_concat.shape))
    print(f'number of neurons left: {total_n_neurons}')

#     preprocessed_feature = preprocess_feature(feature_concat)
#     print(preprocessed_feature.shape)

#     preprocessed_feature_list = []
#     idx = 0
#     for sh in shape_list:
#         n = sh[0] * sh[1]
#         preprocessed_feature_list.append(preprocessed_feature[idx:idx + n].reshape(sh))
#         idx += n

    feature_list = []
    idx = 0
    for sh in shape_list:
        n = sh[0] * sh[1]
        feature_list.append(feature_concat[idx:idx + n].reshape(sh))
        idx += n

    train_shape_list = []
    val_shape_list = []
    test_shape_list = []

    train_bool_list = []
    val_bool_list = []
    test_bool_list = []

    train_trial_list = []
    val_trial_list = []
    test_trial_list = []

    train_feature = []
    val_feature = []
    test_feature = []

    train_output = []
    val_output = []
    test_output = []

    for i in notebook.tqdm(range(len(insertions))):
        try:
            print(session_list[i]['session']['id'])
        except:
            print(session_list[i].tolist()['session']['id'])

        n_trials = feature_list[i].shape[1]
        n_test = int(n_trials * 0.2)
        n_train = int((n_trials - n_test) * 0.8)
        n_val = n_trials - n_train - n_test

        sh = shape_list[i]
        train_shape_list.append((sh[0], n_train,) + sh[-2:])
        val_shape_list.append((sh[0], n_val,) + sh[-2:])
        test_shape_list.append((sh[0], n_test,) + sh[-2:])

        test_idx = rng.choice(np.arange(n_trials), size=n_test, replace=False)
        test_bool = np.zeros(n_trials).astype(bool)
        test_bool[test_idx] = True

        train_idx = rng.choice(np.arange(n_trials)[~test_bool], size=n_train, replace=False)
        train_bool = np.zeros(n_trials).astype(bool)
        train_bool[train_idx] = True

        val_bool = np.zeros(n_trials).astype(bool)
        val_bool[~np.logical_or(test_bool, train_bool)] = True

        train_bool_list.append(train_bool)
        val_bool_list.append(val_bool)
        test_bool_list.append(test_bool)

        train_trial_list.append(trial_number_list[i][train_bool])
        val_trial_list.append(trial_number_list[i][val_bool])
        test_trial_list.append(trial_number_list[i][test_bool])

#         train_feature.append(preprocessed_feature_list[i][:, train_bool].reshape((-1,) + sh[-2:]))
#         val_feature.append(preprocessed_feature_list[i][:, val_bool].reshape((-1,) + sh[-2:]))
#         test_feature.append(preprocessed_feature_list[i][:, test_bool].reshape((-1,) + sh[-2:]))
        train_feature.append(feature_list[i][:, train_bool].reshape((-1,) + sh[-2:]))
        val_feature.append(feature_list[i][:, val_bool].reshape((-1,) + sh[-2:]))
        test_feature.append(feature_list[i][:, test_bool].reshape((-1,) + sh[-2:]))

        train_output.append(output_subset_list[i][:, train_bool].reshape(-1, sh[-2]))
        val_output.append(output_subset_list[i][:, val_bool].reshape(-1, sh[-2]))
        test_output.append(output_subset_list[i][:, test_bool].reshape(-1, sh[-2]))
        
    # standardize data
    train_feature = np.concatenate(train_feature)
    val_feature = np.concatenate(val_feature)
    test_feature = np.concatenate(test_feature)
    train_feature, xyz_stat, dlc_stat, wheel_stat, max_ptp_stat, wf_width_stat = preprocess_feature(train_feature)
    val_feature, _, _, _, _, _ = preprocess_feature(val_feature, xyz_stat, dlc_stat, wheel_stat, max_ptp_stat, wf_width_stat)
    test_feature, _, _, _, _, _ = preprocess_feature(test_feature, xyz_stat, dlc_stat, wheel_stat, max_ptp_stat, wf_width_stat)

    save_path = data_path.joinpath('mtnn_data')
    save_path.mkdir(exist_ok=True, parents=True)

    save_path_train = data_path.joinpath('mtnn_data/train')
    save_path_train.mkdir(exist_ok=True, parents=True)

    save_path_val = data_path.joinpath('mtnn_data/validation')
    save_path_val.mkdir(exist_ok=True, parents=True)

    save_path_test = data_path.joinpath('mtnn_data/test')
    save_path_test.mkdir(exist_ok=True, parents=True)

    np.save(save_path_train.joinpath('shape.npy'), np.asarray(train_shape_list))
    np.save(save_path_val.joinpath('shape.npy'), np.asarray(val_shape_list))
    np.save(save_path_test.joinpath('shape.npy'), np.asarray(test_shape_list))

    np.save(save_path_train.joinpath('bool.npy'), np.asarray(train_bool_list, dtype=object))
    np.save(save_path_val.joinpath('bool.npy'), np.asarray(val_bool_list, dtype=object))
    np.save(save_path_test.joinpath('bool.npy'), np.asarray(test_bool_list, dtype=object))

    np.save(save_path_train.joinpath('trials.npy'), np.asarray(train_trial_list, dtype=object))
    np.save(save_path_val.joinpath('trials.npy'), np.asarray(val_trial_list, dtype=object))
    np.save(save_path_test.joinpath('trials.npy'), np.asarray(test_trial_list, dtype=object))

    np.save(save_path_train.joinpath('feature.npy'), train_feature)
    np.save(save_path_val.joinpath('feature.npy'), val_feature)
    np.save(save_path_test.joinpath('feature.npy'), test_feature)

    np.save(save_path_train.joinpath('output.npy'), np.concatenate(train_output))
    np.save(save_path_val.joinpath('output.npy'), np.concatenate(val_output))
    np.save(save_path_test.joinpath('output.npy'), np.concatenate(test_output))

    np.save(save_path.joinpath('session_info.npy'), np.asarray(session_list))
    np.save(save_path.joinpath('clusters.npy'), np.asarray(cluster_subset_list, dtype=object))


def prepare_glm_and_simulated_data(insertions, one, brain_atlas=None):
    
    download_priors()
    data_load_path = data_path.joinpath('mtnn_data')
    train_trial_ids = np.load(data_load_path.joinpath('train/trials.npy'), allow_pickle=True)
    val_trial_ids = np.load(data_load_path.joinpath('validation/trials.npy'), allow_pickle=True)
    test_trial_ids = np.load(data_load_path.joinpath('test/trials.npy'), allow_pickle=True)
    clusters = np.load(data_load_path.joinpath('clusters.npy'), allow_pickle=True)

    trialsdf_list = []
    prior_list = []
    cluster_list = []
    spk_times_list = []
    clus_list = []
    for i, ins in enumerate(insertions):

        eid = ins['session']['id']
        probe = ins['probe_name']

        trials = one.load_object(eid, 'trials', collection='alf')

        diff1 = trials.firstMovement_times - trials.stimOn_times
        diff2 = trials.feedback_times - trials.firstMovement_times
        t_select1 = np.logical_and(diff1 > 0.0, diff1 < t_before - 0.1)
        t_select2 = np.logical_and(diff2 > 0.0, diff2 < t_after - 0.1)
        keeptrials = np.logical_and(t_select1, t_select2)

        trialsdf = load_trials_df(eid, maxlen=1.5, t_before=0.5, t_after=1.0,
                                wheel_binsize=binwidth, ret_abswheel=False,
                                ret_wheel=True, addtl_types=['firstMovement_times'],
                                one=one, align_event='firstMovement_times', keeptrials=keeptrials)

        trial_idx = np.concatenate([train_trial_ids[i], val_trial_ids[i], test_trial_ids[i]])
        trial_idx = np.sort(trial_idx)
        trialsdf = trialsdf.loc[trial_idx]
#         print(keeptrials.sum(), len(trialsdf))
        trialsdf_list.append(trialsdf)

        pLeft = np.load(save_data_path(figure='fig_mtnn').joinpath('priors', f'prior_{eid}.npy'))
        prior_list.append(pLeft[trial_idx])

        cluster_list.append(clusters[i])

        ba = brain_atlas or AllenAtlas()
        sl = bbone.SpikeSortingLoader(eid=eid, pname=probe, one=one, atlas=ba)
        spikes, _, _ = sl.load_spike_sorting(revision='2024-03-22', enforce_version=False)

        clu_idx = np.isin(spikes.clusters, clusters[i])
        spk_times = spikes.times[clu_idx]
        selected_clus = spikes.clusters[clu_idx]

        spk_times_list.append(spk_times)
        clus_list.append(selected_clus)
#         print(eid, clusters[i])
#         print(eid, np.unique(selected_clus))

    design_list = []
    for i, trialsdf in enumerate(trialsdf_list):
        design = generate_design(trialsdf.copy(), prior_list[i], 0.4, bases, binwidth=binwidth)
        design_list.append(design)

    fit_glm_lists = []
    for i, design in enumerate(design_list):
        nglm = lm.LinearGLM(design, spk_times_list[i], clus_list[i],
                            estimator=RidgeCV(cv=3), binwidth=binwidth)
        nglm.fit(train_idx=np.concatenate([train_trial_ids[i], val_trial_ids[i]]))

    #     pred = GLMPredictor(nglm, spk_times_list[i], clus_list[i], trialsdf_list[i])
    #     for j, unit in enumerate(np.unique(clus_list[i])):
    #         ax = pred.psth_summary('firstMovement_times', unit, t_before=0.5, t_after=1.0)
    #         plt.show()

        fit_glm_lists.append(nglm)

    test_score_list = []
    # glm_leave_one_out = []
    # glm_single_covariate = []
    glm_covs = fit_glm_lists[0].design.covar
    for i, nglm in enumerate(fit_glm_lists):
        score = nglm.score(testinds=test_trial_ids[i])
        test_score_list.append(score)

    #     trlabels = nglm.design.trlabels
    #     train = np.isin(trlabels, train_trial_ids[i]).flatten()
    #     test = np.isin(trlabels, test_trial_ids[i]).flatten()

    #     sfs = mut.SequentialSelector(nglm, train=train, test=test, direction='backward',
    #                                  n_features_to_select=len(nglm.design.covar)-1)
    #     sfs.fit(progress=True)
    #     for clu in sfs.all_scores.index:
    #         sfs.all_scores.loc[clu] = score.loc[clu] - sfs.all_scores.loc[clu]
    #     glm_leave_one_out.append(sfs.all_scores)

    #     sfs = mut.SequentialSelector(nglm, train_trial_ids[i], test_trial_ids[i], direction='forward',
    #                                  n_features_to_select=1)
    #     sfs.fit(progress=True)
    #     glm_single_covariate.append(sfs.all_scores)

    scores = []
    for i in range(len(test_score_list)):
        scores.append(test_score_list[i].loc[cluster_list[i]].to_numpy())
    scores = np.concatenate(scores)
    
    # plt.figure(figsize=(5,5))
    # plt.scatter(scores, scores)
    # plt.axvline(np.median(scores))
    # plt.show()

    glm_score_save_path = data_path.joinpath('glm_data')
    glm_score_save_path.mkdir(exist_ok=True, parents=True)
    np.save(glm_score_save_path.joinpath('glm_scores.npy'), scores)
    
    # Now prepare simulated data
    num_trials = 500 #170 #330
    n_test = int(num_trials*0.2)
    n_train = int((num_trials-n_test)*0.8)
    n_val = num_trials - n_train - n_test

    test_idx = rng.choice(np.arange(num_trials), size=n_test, replace=False)
    test_bool = np.zeros(num_trials).astype(bool)
    test_bool[test_idx] = True

    train_idx = rng.choice(np.arange(num_trials)[~test_bool], size=n_train, replace=False)
    train_bool = np.zeros(num_trials).astype(bool)
    train_bool[train_idx] = True

    val_bool = np.zeros(num_trials).astype(bool)
    val_bool[~np.logical_or(test_bool, train_bool)] = True

    train_idx = np.arange(num_trials)[train_bool]
    val_idx = np.arange(num_trials)[val_bool]
    test_idx = np.arange(num_trials)[test_bool]

    fdb_rt_vals = np.linspace(0.2, 0.7, num=10)
    fdb_rt_probs = np.array([0.15970962, 0.50635209, 0.18693285, 0.0707804, 0.02540835,
                            0.01633394, 0.00907441, 0.00725953, 0.00544465, 0.01270417])

    stim_rt_vals = np.linspace(0.2, 0.4, num=10)
    stim_rt_probs = np.array([0.1324877, 0.10914096, 0.03974338, 0.05596731, 0.14581747, 0.08437234,
                              0.01841735, 0.15889255, 0.16163811, 0.09352284])

    contrastvals = [0.0625, 0.125, 0.25, 1.] + [-0.0625, -0.125, -0.25, -1.]

    unit_id = 0
    mtnn_feature_list = []
    mtnn_output_list = []

    simulated_feature_list = []
    simulated_output_list = []
    simulated_glm_leave_one_out = []
    simulated_glm_scores = []

    scales_dict = {}

    clus_to_remove = defaultdict(list)
    for i, ins in enumerate(insertions):
        eid = ins['session']['id']
        print('processing session {}'.format(eid))

        trialsdf = trialsdf_list[i]
        nglm = fit_glm_lists[i]
        clus = cluster_list[i]
        weights = nglm.combine_weights()
        intercepts = nglm.intercepts

        wheeltraces = trialsdf.wheel_velocity.to_list()
        if len(wheeltraces) < num_trials: # use wheel traces from other sessions
            for j in range(len(insertions)):
                if i == j:
                    continue
                new_wheeltraces = trialsdf_list[j].wheel_velocity.to_list()
                wheeltraces.extend(new_wheeltraces[:num_trials-len(wheeltraces)])
                
                if len(wheeltraces) >= num_trials:
                    break
        if len(wheeltraces) > num_trials:
            wheeltraces = wheeltraces[:num_trials]
        firstMovement_times = np.ones(num_trials) * 0.5
        stimtimes = rng.choice(stim_rt_vals, size=num_trials, p=stim_rt_probs) \
            + rng.normal(size=num_trials) * 0.05
        fdbktimes = rng.choice(fdb_rt_vals, size=num_trials, p=fdb_rt_probs) \
            + firstMovement_times + rng.normal(size=num_trials) * 0.05
        priors = prior_list[i]
        if priors.shape[0] < num_trials:
            for j in range(len(insertions)):
                if i == j:
                    continue
                new_priors = prior_list[j]
                priors = np.concatenate((priors, new_priors[:num_trials-priors.shape[0]]))
                if priors.shape[0] >= num_trials:
                    break
        if priors.shape[0] > num_trials:
            priors = priors[:num_trials]
        prev_priors = np.pad(priors, (1, 0), constant_values=0)[:-1]
        contrasts = rng.choice(contrastvals, replace=True, size=num_trials)
        feedbacktypes = rng.choice([-1, 1], size=num_trials, p=[0.1, 0.9])
        wheelmoves = rng.choice(np.arange(len(wheeltraces)), size=num_trials)

        session_simulated_spkidx_list = []
        session_simulated_feature_list = []

        print(f'total number of units: {len(clus)}')

        for j, clu in enumerate(clus):
            
            try:
                stimkernL = weights['stimonL'].loc[clu].to_numpy() * (1/binwidth)
                stimkernR = weights['stimonR'].loc[clu].to_numpy() * (1/binwidth)
                fdbkkern1 = weights['correct'].loc[clu].to_numpy() * (1/binwidth)
                fdbkkern2 = weights['incorrect'].loc[clu].to_numpy() * (1/binwidth)
                fmovekern = weights['fmove'].loc[clu].to_numpy() * (1/binwidth)
                wheelkern = weights['wheel'].loc[clu].to_numpy() * (1/binwidth)
                intercept = intercepts.loc[clu] * (1/binwidth) 
            except:
                clus_to_remove[eid].append(clu)
                continue

            ret = simulate_cell((stimkernL,stimkernR), (fdbkkern1,fdbkkern2),
                                fmovekern, wheelkern,
                                wheeltraces,
                                stimtimes, fdbktimes, feedbacktypes, firstMovement_times,
                                priors, prev_priors, contrasts,
                                wheelmoves, pgain=5.0, gain=10.0,
                                num_trials=num_trials, linear=True, ret_raw=False)
            trialspikes, contrasts, priors, stimtimes, fdbktimes, fmovetimes, feedbacktypes, trialwheel = ret[:-2]
            adj_spkt, new_trialsdf = concat_simcell_data(trialspikes, contrasts, priors,
                                                     stimtimes, fdbktimes, fmovetimes,
                                                     feedbacktypes, trialwheel)
            sess_clu = np.ones_like(adj_spkt, dtype=int) * unit_id

            if j == 0:
                design = generate_design(new_trialsdf.copy(), priors, 0.4, bases,
                                         prior_last=prev_priors, binwidth=binwidth)
                feature = to_mtnn_form(new_trialsdf)
            feature = feature.copy()
            feature[:,:,0] = unit_id
            session_simulated_feature_list.append(feature[None])

            nglm = lm.LinearGLM(design, adj_spkt, sess_clu,
                            estimator=RidgeCV(alphas=alphas, cv=5), binwidth=binwidth)
            nglm.fit(train_idx=np.concatenate([train_idx, val_idx]))

#             pred = GLMPredictor(nglm, adj_spkt, sess_clu, new_trialsdf, np.arange(num_trials))
#             for k, unit in enumerate(np.unique(sess_clu)):
#                 ax = pred.psth_summary('firstMovement_times', unit, t_before=0.5, t_after=1.0)
#                 plt.show()

            score = nglm.score(testinds=test_idx)
            simulated_glm_scores.append(score)
            
#             sfs2 = mut.SequentialSelector(nglm, train=np.concatenate([train_idx, val_idx]), test=test_idx, direction='forward',
#                                          n_features_to_select=len(nglm.design.covar))
#             sfs2.fit(progress=False)

            sfs = mut.SequentialSelector(nglm, train=np.concatenate([train_idx, val_idx]), test=test_idx, direction='backward',
                                         n_features_to_select=len(nglm.design.covar)-1)
            sfs.fit(full_scores=True, progress=False)

            sfs.full_scores_test_.loc[unit_id,0] = score.loc[unit_id] - sfs.full_scores_test_.loc[unit_id,0]
            simulated_glm_leave_one_out.append(sfs.full_scores_test_.loc[unit_id,0].to_numpy()[None].astype(np.float32))

            unit_id += 1

            nbins = int(1.5/binwidth)
            raster = np.zeros((len(new_trialsdf), nbins))
            for trial in range(len(new_trialsdf)):
                for n in range(nbins):
                    idx = np.logical_and(adj_spkt>=1.5*trial+binwidth*n, adj_spkt<1.5*trial+binwidth*(n+1))
                    raster[trial,n] = idx.astype(int).sum() / binwidth

#             plt.figure(figsize=(8,2))
# #             plt.plot(raster.mean(0), color='k')
#             plt.imshow(raster, aspect='auto', interpolation='none')
#             plt.show()

            session_simulated_spkidx_list.append(raster[None])
        simulated_output_list.append(np.concatenate(session_simulated_spkidx_list, axis=0))
        simulated_feature_list.append(np.concatenate(session_simulated_feature_list, axis=0))
    print(clus_to_remove)
    
    simulated_glm_leave_one_out = np.concatenate(simulated_glm_leave_one_out, axis=0)
    simulated_glm_scores = pd.concat(simulated_glm_scores)

    shape_list = []
    feature_concat = []
    for i in notebook.tqdm(range(len(simulated_feature_list))):
        shape_list.append(simulated_feature_list[i].shape)
        feature_concat.append(simulated_feature_list[i].reshape((-1,)+simulated_feature_list[i].shape[-2:]))
    feature_concat = np.concatenate(feature_concat)
    print('feature_concat shape: {}'.format(feature_concat.shape))

    wheel_min = feature_concat[:,:,8].min()
    wheel_max = feature_concat[:,:,8].max()
    feature_concat[:,:,8] = -1 + 2*(feature_concat[:,:,8] - wheel_min) / (wheel_max - wheel_min)

    preprocessed_feature_list = []
    idx = 0
    for sh in shape_list:
        n = sh[0]*sh[1]
        preprocessed_feature_list.append(feature_concat[idx:idx+n].reshape(sh))
        idx += n

    train_shape_list = []
    val_shape_list = []
    test_shape_list = []

    train_bool_list = []
    val_bool_list = []
    test_bool_list = []

    train_trial_list = []
    val_trial_list = []
    test_trial_list = []

    train_feature = []
    val_feature = []
    test_feature = []

    train_output = []
    val_output = []
    test_output = []

    for i in notebook.tqdm(range(len(preprocessed_feature_list))):

        sh = shape_list[i]

        train_shape_list.append((sh[0],n_train,)+sh[-2:])
        val_shape_list.append((sh[0],n_val,)+sh[-2:])
        test_shape_list.append((sh[0],n_test,)+sh[-2:])

        train_bool_list.append(train_bool)
        val_bool_list.append(val_bool)
        test_bool_list.append(test_bool)

        train_trial_list.append(train_idx)
        val_trial_list.append(val_idx)
        test_trial_list.append(test_idx)

        train_feature.append(preprocessed_feature_list[i][:,train_bool].reshape((-1,)+sh[-2:]))
        val_feature.append(preprocessed_feature_list[i][:,val_bool].reshape((-1,)+sh[-2:]))
        test_feature.append(preprocessed_feature_list[i][:,test_bool].reshape((-1,)+sh[-2:]))

        train_output.append(simulated_output_list[i][:,train_bool].reshape(-1, sh[-2]))
        val_output.append(simulated_output_list[i][:,val_bool].reshape(-1, sh[-2]))
        test_output.append(simulated_output_list[i][:,test_bool].reshape(-1, sh[-2]))

    save_path = data_path.joinpath('simulated_data')
    save_path.mkdir(exist_ok=True, parents=True)

    save_path_train = data_path.joinpath('simulated_data/train')
    save_path_train.mkdir(exist_ok=True, parents=True)

    save_path_val = data_path.joinpath('simulated_data/validation')
    save_path_val.mkdir(exist_ok=True, parents=True)

    save_path_test = data_path.joinpath('simulated_data/test')
    save_path_test.mkdir(exist_ok=True, parents=True)

    np.save(save_path_train.joinpath('shape.npy'), np.asarray(train_shape_list))
    np.save(save_path_val.joinpath('shape.npy'), np.asarray(val_shape_list))
    np.save(save_path_test.joinpath('shape.npy'), np.asarray(test_shape_list))

    np.save(save_path_train.joinpath('bool.npy'), np.asarray(train_bool_list, dtype=object))
    np.save(save_path_val.joinpath('bool.npy'), np.asarray(val_bool_list, dtype=object))
    np.save(save_path_test.joinpath('bool.npy'), np.asarray(test_bool_list, dtype=object))

    np.save(save_path_train.joinpath('trials.npy'), np.asarray(train_trial_list, dtype=object))
    np.save(save_path_val.joinpath('trials.npy'), np.asarray(val_trial_list, dtype=object))
    np.save(save_path_test.joinpath('trials.npy'), np.asarray(test_trial_list, dtype=object))

    np.save(save_path_train.joinpath('feature.npy'), np.concatenate(train_feature))
    np.save(save_path_val.joinpath('feature.npy'), np.concatenate(val_feature))
    np.save(save_path_test.joinpath('feature.npy'), np.concatenate(test_feature))

    np.save(save_path_train.joinpath('output.npy'), np.concatenate(train_output))
    np.save(save_path_val.joinpath('output.npy'), np.concatenate(val_output))
    np.save(save_path_test.joinpath('output.npy'), np.concatenate(test_output))

    np.save(save_path.joinpath('glm_scores.npy'), simulated_glm_scores)
    np.save(save_path.joinpath('glm_leave_one_out.npy'), simulated_glm_leave_one_out)

    
def prepare_glm_data_full_mtnn_cov(insertions, one, brain_atlas=None):

    #download_priors()
    data_load_path = data_path.joinpath('mtnn_data')
    train_trial_ids = np.load(data_load_path.joinpath('train/trials.npy'), allow_pickle=True)
    val_trial_ids = np.load(data_load_path.joinpath('validation/trials.npy'), allow_pickle=True)
    test_trial_ids = np.load(data_load_path.joinpath('test/trials.npy'), allow_pickle=True)
    clusters = np.load(data_load_path.joinpath('clusters.npy'), allow_pickle=True)

    trialsdf_list = []
    prior_list = []
    cluster_list = []
    spk_times_list = []
    clus_list = []
    for i, ins in notebook.tqdm(enumerate(insertions)):

        eid = ins['session']['id']
        probe = ins['probe_name']

        trials = one.load_object(eid, 'trials', collection='alf')

        diff1 = trials.firstMovement_times - trials.stimOn_times
        diff2 = trials.feedback_times - trials.firstMovement_times
        t_select1 = np.logical_and(diff1 > 0.0, diff1 < t_before - 0.1)
        t_select2 = np.logical_and(diff2 > 0.0, diff2 < t_after - 0.1)
        keeptrials = np.logical_and(t_select1, t_select2)

        trialsdf = load_trials_df(eid, maxlen=1.5, t_before=0.5, t_after=1.0,
                                wheel_binsize=binwidth, ret_abswheel=False,
                                ret_wheel=False, addtl_types=['response_times', 'firstMovement_times'],
                                one=one, align_event='firstMovement_times', keeptrials=keeptrials)

        trial_idx = np.concatenate([train_trial_ids[i], val_trial_ids[i], test_trial_ids[i]])
        trial_idx = np.sort(trial_idx)
        trialsdf = trialsdf.loc[trial_idx]

        # load feature
        feature = np.load(data_path.joinpath('original_data').joinpath(f'{eid}_feature.npy'), allow_pickle=True)

        # add wheel
        x = feature[0,:,:,wheel_offset]
        x = 2 * (x - x.min()) / (x.max() - x.min()) - 1
        wheel_velocity_pd = pd.Series(list(x), index = trialsdf.index)
        trialsdf['wheel_velocity'] = wheel_velocity_pd.values

        # add paw
        x = feature[0,:,:,paw_offset]
        x = (x - x.min()) / (x.max() - x.min())
        paw_speed_pd = pd.Series(list(x), index = trialsdf.index)
        trialsdf['paw_speed'] = paw_speed_pd.values

        # add nose
        x = feature[0,:,:,nose_offset]
        x = (x - x.min()) / (x.max() - x.min())
        nose_speed_pd = pd.Series(list(x), index = trialsdf.index)
        trialsdf['nose_speed'] = nose_speed_pd.values

        # add pupil
        x = feature[0,:,:,pupil_offset]
        x = (x - x.min()) / (x.max() - x.min())
        pupil_dia_pd = pd.Series(list(x), index = trialsdf.index)
        trialsdf['pupil_diameter'] = pupil_dia_pd.values

        # add me
        x = feature[0,:,:,left_me_offset]
        x = (x - x.min()) / (x.max() - x.min())
        left_me_pd = pd.Series(list(x), index = trialsdf.index)
        trialsdf['left_me'] = left_me_pd.values

        # add lick
        x = feature[0,:,:,lick_offset]
        x = (x - x.min()) / (x.max() - x.min() + 1e-6)
        lick_pd = pd.Series(list(x), index = trialsdf.index)
        trialsdf['lick'] = lick_pd.values

        trialsdf_list.append(trialsdf)

        pLeft = np.load(save_data_path(figure='fig_mtnn').joinpath('priors', f'prior_{eid}.npy'))
        prior_list.append(pLeft[trial_idx])

        cluster_list.append(clusters[i])

        ba = brain_atlas or AllenAtlas()
        sl = bbone.SpikeSortingLoader(eid=eid, pname=probe, one=one, atlas=ba)
        spikes, _, _ = sl.load_spike_sorting(revision='2024-03-22', enforce_version=False)

        clu_idx = np.isin(spikes.clusters, clusters[i])
        spk_times = spikes.times[clu_idx]
        selected_clus = spikes.clusters[clu_idx]

        spk_times_list.append(spk_times)
        clus_list.append(selected_clus)

    design_list = []
    for i, trialsdf in enumerate(trialsdf_list):
        design = generate_design_full_mtnn_cov(trialsdf.copy(), prior_list[i], 0.4, 
                                               bases_full_mtnn_cov, duration=1.5,
                                               binwidth=binwidth)
        design_list.append(design)

    fit_glm_lists = []
    for i, design in enumerate(design_list):
        nglm = lm.LinearGLM(design, spk_times_list[i], clus_list[i],
                            estimator=RidgeCV(cv=3), binwidth=binwidth)
        nglm.fit(train_idx=np.concatenate([train_trial_ids[i], val_trial_ids[i]]))

        fit_glm_lists.append(nglm)

    test_score_list = []
    glm_covs = fit_glm_lists[0].design.covar
    for i, nglm in enumerate(fit_glm_lists):
        score = nglm.score(testinds=test_trial_ids[i])
        test_score_list.append(score)

    scores = []
    for i in range(len(test_score_list)):
        scores.append(test_score_list[i].loc[cluster_list[i]].to_numpy())
    scores = np.concatenate(scores)

    glm_score_save_path = data_path.joinpath('glm_data')
    glm_score_save_path.mkdir(exist_ok=True, parents=True)
    np.save(glm_score_save_path.joinpath('glm_scores_full_mtnn_cov.npy'), scores)

if __name__ == '__main__':
    one = ONE()
    one.record_loaded = True
    prepare_data(one)
    save_dataset_info(one, figure='fig_mtnn')

from one.api import ONE
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
import brainbox.modeling.design_matrix as dm
import brainbox.modeling.linear as lm
import brainbox.modeling.utils as mut
import brainbox.io.one as bbone
from brainbox.modeling.design_matrix import convbasis

from numpy.random import uniform, normal

from sklearn.linear_model import RidgeCV

import matplotlib.pyplot as plt

from utils import *

from reproducible_ephys_functions import save_data_path

import matplotlib.pyplot as plt
from brainbox.modeling import linear
from brainbox.modeling import poisson
from brainbox.plot import peri_event_time_histogram

binwidth = 0.05
t_before = 0.5
t_after = 1.0
align_event = 'firstMovement_times'

def tmp_binf(t):
    return np.ceil(t / binwidth).astype(int)

bases = {
    'stim': mut.raised_cosine(0.4, 5, tmp_binf),
    'feedback': mut.raised_cosine(0.4, 5, tmp_binf),
    'wheel': mut.raised_cosine(0.3, 3, tmp_binf),
    'fmove': mut.raised_cosine(0.2, 3, tmp_binf),
}

def generate_design(trialsdf, prior, t_before, bases, prior_last=None,
                    iti_prior=[-0.4, -0.1], fmove_offset=-0.4, wheel_offset=-0.4,
                    contnorm=5., binwidth=0.05, reduce_wheel_dim=True):
    """
    Generate GLM design matrix object
    Parameters
    ----------
    trialsdf : pd.DataFrame
        Trials dataframe with trial timings in absolute (since session start) time
    prior : array-like
        Vector containing the prior estimate or true prior for each trial. Must be same length as
        trialsdf.
    t_before : float
        Time, in seconds, before stimulus onset that was used to define trial_start in trialsdf
    bases : dict
        Dictionary of basis functions for each regressor. Needs keys 'stim', 'feedback', 'fmove',
        (first movement) and 'wheel'.
    iti_prior : list, optional
        Two element list defining bounds on which step function for ITI prior is
        applied, by default [-0.4, -0.1]
    contnorm : float, optional
        Normalization factor for contrast, by default 5.
    binwidth : float, optional
        Size of bins to use for design matrix, in seconds, by default 0.02
    """
    trialsdf['adj_contrastL'] = np.tanh(contnorm * trialsdf['contrastLeft']) / np.tanh(contnorm)
    trialsdf['adj_contrastR'] = np.tanh(contnorm * trialsdf['contrastRight']) / np.tanh(contnorm)
    trialsdf['prior'] = prior
    if prior_last is None:
        trialsdf['prior_last'] = pd.Series(np.roll(trialsdf['prior'], 1), index=trialsdf.index)
    else:
        trialsdf['prior_last'] = prior_last

    vartypes = {'choice': 'value',
                'response_times': 'timing',
                'probabilityLeft': 'value',
                'feedbackType': 'value',
                'feedback_times': 'timing',
                'contrastLeft': 'value',
                'adj_contrastL': 'value',
                'contrastRight': 'value',
                'adj_contrastR': 'value',
                'goCue_times': 'timing',
                'stimOn_times': 'timing',
                'trial_start': 'timing',
                'trial_end': 'timing',
                'prior': 'value',
                'prior_last': 'value',
                'wheel_velocity': 'continuous',
                'firstMovement_times': 'timing'}

    def stepfunc_prestim(row):
        stepvec = np.zeros(design.binf(row.duration))
        diff = row.stimOn_times - row.trial_start
        stimOn_bin = design.binf(diff)
        stepvec[0:stimOn_bin-1] = row.prior_last
        return stepvec

    def stepfunc_poststim(row):
        zerovec = np.zeros(design.binf(row.duration))
        currtr_start = design.binf(row.stimOn_times + 0.1)
        currtr_end = design.binf(row.feedback_times)
        zerovec[currtr_start:currtr_end] = row.prior_last
        zerovec[currtr_end:] = row.prior
        return zerovec

    design = dm.DesignMatrix(trialsdf, vartypes, binwidth=binwidth)
#     stepbounds = [design.binf(t_before + iti_prior[0]), design.binf(t_before + iti_prior[1])]
#     print(stepbounds)

    design.add_covariate_timing('stimonL', 'stimOn_times', bases['stim'],
                                cond=lambda tr: np.isfinite(tr.contrastLeft),
                                deltaval='adj_contrastL',
                                desc='Kernel conditioned on L stimulus onset')
    design.add_covariate_timing('stimonR', 'stimOn_times', bases['stim'],
                                cond=lambda tr: np.isfinite(tr.contrastRight),
                                deltaval='adj_contrastR',
                                desc='Kernel conditioned on R stimulus onset')
    design.add_covariate_timing('correct', 'feedback_times', bases['feedback'],
                                cond=lambda tr: tr.feedbackType == 1,
                                desc='Kernel conditioned on correct feedback')
    design.add_covariate_timing('incorrect', 'feedback_times', bases['feedback'],
                                cond=lambda tr: tr.feedbackType == -1,
                                desc='Kernel conditioned on incorrect feedback')
    design.add_covariate_timing('fmove', 'firstMovement_times', bases['fmove'],
                                offset=fmove_offset,
                                desc='Lead up to first movement')
    design.add_covariate_raw('pLeft', stepfunc_prestim,
                             desc='Step function on prior estimate')
    design.add_covariate_raw('pLeft_tr', stepfunc_poststim,
                             desc='Step function on post-stimulus prior')

    design.add_covariate('wheel', trialsdf['wheel_velocity'], bases['wheel'], wheel_offset)
    design.compile_design_matrix()

    if reduce_wheel_dim:
        _, s, v = np.linalg.svd(design[:, design.covar['wheel']['dmcol_idx']],
                                full_matrices=False)
        variances = s**2 / (s**2).sum()
        n_keep = np.argwhere(np.cumsum(variances) >= 0.9999)[0, 0]
        wheelcols = design[:, design.covar['wheel']['dmcol_idx']]
        reduced = wheelcols @ v[:n_keep].T
        bases_reduced = bases['wheel'] @ v[:n_keep].T
        keepcols = ~np.isin(np.arange(design.dm.shape[1]), design.covar['wheel']['dmcol_idx'])
        basedm = design[:, keepcols]
        design.dm = np.hstack([basedm, reduced])
        design.covar['wheel']['dmcol_idx'] = design.covar['wheel']['dmcol_idx'][:n_keep]
        design.covar['wheel']['bases'] = bases_reduced

    print('Condition of design matrix:', np.linalg.cond(design.dm))
    return design

def predict(nglm, targ_regressors=None, trials=None, retlab=False, incl_bias=True):
    if trials is None:
        trials = nglm.design.trialsdf.index
    if targ_regressors is None:
        targ_regressors = nglm.design.covar.keys()
    dmcols = np.hstack([nglm.design.covar[r]['dmcol_idx'] for r in targ_regressors])
    dmcols = np.sort(dmcols)
    trlabels = nglm.design.trlabels
    trfilter = np.isin(trlabels, trials).flatten()
    w = nglm.coefs
    b = nglm.intercepts
    dm = nglm.design.dm[trfilter, :][:, dmcols]
    if type(nglm) == poisson.PoissonGLM:
        link = np.exp
    elif type(nglm) == linear.LinearGLM:
        def link(x):
            return x
    else:
        raise TypeError('nglm must be poisson or linear')
    if incl_bias:
        pred = {cell: link(dm @ w.loc[cell][dmcols] + b.loc[cell]) for cell in w.index}
    else:
        pred = {cell: link(dm @ w.loc[cell][dmcols]) for cell in w.index}
    # if type(nglm) == LinearGLM:
    #     for cell in pred:
    #         cellind = np.argwhere(nglm.clu_ids == cell)[0][0]
    #         pred[cell] += np.mean(nglm.binnedspikes[:, cellind])
    if not retlab:
        return pred
    else:
        return pred, trlabels[trfilter].flatten()

def pred_psth(nglm, align_time, t_before, t_after, targ_regressors=None, trials=None,
              incl_bias=True):
    if trials is None:
        trials = nglm.design.trialsdf.index
    times = nglm.design.trialsdf[align_time].apply(nglm.binf)
    tbef_bin = nglm.binf(t_before)
    taft_bin = nglm.binf(t_after)
    pred, labels = predict(nglm, targ_regressors, trials, retlab=True, incl_bias=incl_bias)
    t_inds = [np.searchsorted(labels, tr) + times[tr] for tr in trials]
    winds = [(t - tbef_bin, t + taft_bin) for t in t_inds]
    psths = {}
    for cell in pred.keys():
        cellpred = pred[cell]
        windarr = np.vstack([cellpred[w[0]:w[1]] for w in winds])
        psths[cell] = (np.mean(windarr, axis=0) / nglm.binwidth,
                       np.std(windarr, axis=0) / nglm.binwidth)
    return psths

class GLMPredictor:
    def __init__(self, nglm, spk_t, spk_clu, trialsdf, trial_ids):
        self.covar = list(nglm.design.covar.keys())
        self.nglm = nglm
        self.binnedspikes = nglm.binnedspikes
        self.design = nglm.design
        self.spk_t = spk_t
        self.spk_clu = spk_clu
        self.trials = trial_ids#nglm.design.trialsdf.index
        self.trialsdf = trialsdf#nglm.design.trialsdf #maybe not best way to do this
        self.full_psths = {}
        self.cov_psths = {}
        self.combweights = nglm.combine_weights()

    def psth_summary(self, align_time, unit, t_before=0.1, t_after=0.6, ax=None):
        if ax is None:
            fig, ax = plt.subplots(3, 1, figsize=(8, 12))

        times = self.trialsdf.loc[self.trials, align_time] #
        peri_event_time_histogram(self.spk_t, self.spk_clu,
                                  times,
                                  unit, t_before, t_after, bin_size=self.nglm.binwidth,
                                  error_bars='sem', ax=ax[0], smoothing=0.0)
        keytuple = (align_time, t_before, t_after)
        if keytuple not in self.full_psths:
            self.full_psths[keytuple] = pred_psth(self.nglm, align_time, t_before, t_after,
                                                  trials=self.trials)
            self.cov_psths[keytuple] = {}
            tmp = self.cov_psths[keytuple]
            for cov in self.covar:
                tmp[cov] = pred_psth(self.nglm, align_time, t_before, t_after,
                                     targ_regressors=[cov], trials=self.trials,
                                     incl_bias=False)
                ax[2].plot(self.combweights[cov].loc[unit])
        x = np.arange(-t_before, t_after, self.nglm.binwidth) + 0.01
        ax[0].plot(x, self.full_psths[keytuple][unit][0], label='Model prediction', color='r')
        ax[0].legend()
        for cov in self.covar:
            ax[1].plot(x, self.cov_psths[keytuple][cov][unit][0], label=cov)
        ax[1].set_title('Individual component contributions')
        ax[1].legend()
        if hasattr(self.nglm, 'clu_regions'):
            unitregion = self.nglm.clu_regions[unit]
            plt.suptitle(f'Unit {unit} from region {unitregion}')
        else:
            plt.suptitle(f'Unit {unit}')
        plt.tight_layout()
        return ax
    
def simulate_cell(stimkerns, fdbkkerns, fmovekern, wheelkern, wheeltraces,
                  stimtimes, fdbktimes, feedbacktypes, fmovetimes, priors, prev_priors, contrasts,
                  wheelmoves, pgain=5.0, gain=15.0, num_trials=300, linear=True, ret_raw=False):

    trialspikes = []
    trialrates = []
    trialwheel = []
    if ret_raw:
        trialcont = []
    trialrange = range(num_trials)
    zipiter = zip(trialrange, stimtimes, fdbktimes, fmovetimes, priors, prev_priors,
                  contrasts, feedbacktypes, wheelmoves)
    for i, start, end, fmove, prior, prev_prior, contrast, feedbacktype, wheelchoice in zipiter:
        trial_len = int(1.5 / binwidth)
        stimarr = np.zeros(trial_len)
        fdbkarr = np.zeros(trial_len)
        fmovekarr = np.zeros(trial_len)
        stimarr[int(np.ceil(start / binwidth))] = 1
        fdbkarr[int(np.ceil(end / binwidth))] = 1
        fmovekarr[int(np.ceil(fmove / binwidth))] = 1
        stimkern = stimkerns[0] if contrast > 0 else stimkerns[1]
        fdbkkern = fdbkkerns[0] if feedbacktype == 1 else fdbkkerns[1]
        stimarr = np.convolve(stimkern, stimarr)[:trial_len]
        fdbkarr = np.convolve(fdbkkern, fdbkarr)[:trial_len]
        fmovearr = np.convolve(fmovekern, fmovekarr)[:trial_len]
        fdbkind = int(np.ceil(end / binwidth))

        wheel = wheeltraces[wheelchoice].copy()
        lendiff = trial_len - wheel.shape[0]
        if lendiff >= 0:
            wheel = np.pad(wheel, (0, lendiff), constant_values=0)
        else:
            wheel = wheel[:lendiff]
        wheelinterp = interp1d(np.arange(len(wheel)) * binwidth,
                               wheel, fill_value='extrapolate')
        wheelnew = wheelinterp(np.arange(trial_len) * binwidth)
        wheelarr = convbasis(wheelnew.reshape(-1, 1),
                             wheelkern.reshape(-1, 1),
                             offset=-np.ceil(0.3 / binwidth).astype(int)).flatten()
        
        priorarr = np.array([prev_prior] * fdbkind +
                            [prior] * (trial_len - fdbkind))
        priorarr = pgain * priorarr
        kernsum = priorarr + stimarr + fdbkarr + fmovearr + wheelarr

        if not linear:
            ratevals = np.exp(kernsum + gain) * binwidth
            spikecounts = np.random.poisson(ratevals)
        else:
            ratevals = (kernsum + gain) * binwidth
            contspikecounts = np.random.normal(
                loc=ratevals, scale=gain * binwidth)
            spikecounts = np.round(contspikecounts).astype(int)
        if ret_raw:
            trialcont.append(contspikecounts)
        spike_times = []

        noisevals = uniform(low=0, high=binwidth - 1e-8,
                            size=np.max(spikecounts))
        for i in np.nonzero(spikecounts)[0]:
            curr_t = i * binwidth
            for j in range(spikecounts[i]):
                jitterspike = curr_t + noisevals[j]
                if jitterspike < 0:
                    jitterspike = 0
                spike_times.append(jitterspike)
        trialspikes.append(spike_times)
        trialrates.append(ratevals)
        trialwheel.append(wheel)
    retlist = [trialspikes, contrasts, priors, stimtimes,
               fdbktimes, fmovetimes, feedbacktypes, trialwheel, trialrates,
               trialcont if ret_raw else None]
    return retlist

def concat_simcell_data(trialspikes, contrasts, priors, stimtimes, fdbktimes, fmovetimes, feedbacktypes,
                        trialwheel):
    trialsdf = pd.DataFrame()
    trialends = np.cumsum(fmovetimes + 1.0)
    trialends = np.pad(trialends, (1,0), constant_values=0)
    cat_stimtimes = np.array(
        [trialends[i] + st for i, st in enumerate(stimtimes)])
    cat_fdbktimes = np.array(
        [trialends[i] + ft for i, ft in enumerate(fdbktimes)])
    cat_fmovetimes = np.array(
        [trialends[i] + fmt for i, fmt in enumerate(fmovetimes)])
    contrastLeft = np.empty_like(contrasts)
    contrastRight = np.empty_like(contrasts)
    contrastLeft[:] = np.NaN
    contrastRight[:] = np.NaN
    
    contrastLeft[np.where(contrasts>0)] = contrasts[np.where(contrasts>0)]
    contrastRight[np.where(contrasts<0)] = contrasts[np.where(contrasts<0)]
    
    trialsdf['contrastLeft'] = contrastLeft
    trialsdf['contrastRight'] = contrastRight
    trialsdf['prior'] = priors
    trialsdf['prior_last'] = np.pad(priors[1:], (0, 1), constant_values=0)
    trialsdf['trial_start'] = trialends[:-1]
    trialsdf['trial_end'] = trialends[1:]
    trialsdf['stimOn_times'] = cat_stimtimes
    trialsdf['feedback_times'] = cat_fdbktimes
    trialsdf['firstMovement_times'] = cat_fmovetimes
    trialsdf['feedbackType'] = feedbacktypes
    trialsdf['wheel_velocity'] = trialwheel

    indices = trialsdf.index
    adj_spkt = np.hstack([trialsdf.loc[i].trial_start + np.array(t)
                          for i, t in zip(indices, trialspikes)])
    return np.sort(adj_spkt), trialsdf

def to_mtnn_form(trialsdf, trial_len=30, nfeats=8, binwidth=0.05):
    feature = np.zeros((len(trialsdf), trial_len, nfeats+1))
    for i in range(len(trialsdf)):
        stimon_bin = (trialsdf.loc[i]['stimOn_times']-trialsdf.loc[i]['trial_start'])/binwidth
        stimon_bin = np.floor(stimon_bin).astype(int)
        
        feedback_bin = (trialsdf.loc[i]['feedback_times']-trialsdf.loc[i]['trial_start'])/binwidth
        feedback_bin = np.floor(feedback_bin).astype(int)
        
        fmv_bin = (trialsdf.loc[i]['firstMovement_times']-trialsdf.loc[i]['trial_start'])/binwidth
        fmv_bin = np.floor(fmv_bin).astype(int)
        
        if np.isnan(trialsdf.loc[i]['contrastRight']):
            feature[i,stimon_bin,1] = np.abs(trialsdf.loc[i]['contrastLeft'])
        else:
            feature[i,stimon_bin,2] = np.abs(trialsdf.loc[i]['contrastRight'])
            
        if trialsdf.loc[i]['feedbackType'] == -1:
            feature[i,feedback_bin,3] = 1
        else:
            feature[i,feedback_bin,4] = 1
        
        feature[i,fmv_bin,5] = 1
        
        feature[i,:,6] = trialsdf.loc[i]['prior']
        feature[i,:,7] = trialsdf.loc[i]['prior_last']
        feature[i,:,8] = trialsdf.loc[i]['wheel_velocity']
    
    return feature
    
    
########main##########

one = ONE()

mtnn_eids = get_mtnn_eids()
print(list(mtnn_eids.keys()))

trajs = get_traj(mtnn_eids)

data_load_path = save_data_path(figure='figure8').joinpath('mtnn_data')

train_trial_ids = np.load(data_load_path.joinpath('train/trials.npy'), allow_pickle=True)
val_trial_ids = np.load(data_load_path.joinpath('validation/trials.npy'), allow_pickle=True)
test_trial_ids = np.load(data_load_path.joinpath('test/trials.npy'), allow_pickle=True)

clusters = np.load(data_load_path.joinpath('clusters.npy'), allow_pickle=True)

trialsdf_list = []
prior_list = []
cluster_list = []
spk_times_list = []
clus_list = []
for i, eid in enumerate(mtnn_eids):
    
    trials = one.load_object(eid, 'trials', collection='alf')
    
    diff1 = trials.firstMovement_times - trials.stimOn_times
    diff2 = trials.feedback_times - trials.firstMovement_times
    t_select1 = np.logical_and(diff1 > 0.0, diff1 < t_before-0.1)
    t_select2 = np.logical_and(diff2 > 0.0, diff2 < t_after-0.1)
    keeptrials = np.logical_and(t_select1, t_select2)
    
    trialsdf = bbone.load_trials_df(eid,
                                    maxlen=1.5, t_before=0.5, t_after=1.0,
                                    wheel_binsize=binwidth, ret_abswheel=False,
                                    ret_wheel=True, addtl_types=['firstMovement_times'],
                                    one=one, align_event='firstMovement_times', keeptrials=keeptrials)
    
    trial_idx = np.concatenate([train_trial_ids[i], val_trial_ids[i], test_trial_ids[i]])
    trial_idx = np.sort(trial_idx)
    trialsdf = trialsdf.loc[trial_idx]
    trialsdf_list.append(trialsdf)
    
    pLeft = np.load('./priors/prior_{}.npy'.format(eid))
    prior_list.append(pLeft[trial_idx])
    
    cluster_list.append(clusters[i])
    
    traj = trajs[i]
    probe = traj['probe_name']
    spikes, clus, channels = bbone.load_spike_sorting_with_channel(eid, 
                                                                   one=one, 
                                                                   probe=probe, 
                                                                   spike_sorter='pykilosort')
    spikes = spikes[probe]
    clus = clus[probe]
    channels = channels[probe]
    
    clu_idx = np.isin(spikes.clusters, clusters[i])
    spk_times = spikes.times[clu_idx]
    selected_clus = spikes.clusters[clu_idx]
    
    spk_times_list.append(spk_times)
    clus_list.append(selected_clus)
    
design_list = []
for i, trialsdf in enumerate(trialsdf_list):
    design = generate_design(trialsdf.copy(), prior_list[i], 0.4, bases, binwidth=binwidth)
    design_list.append(design)

fit_glm_lists = []
for i, design in enumerate(design_list):
    nglm = lm.LinearGLM(design, spk_times_list[i], clus_list[i], 
                        estimator=RidgeCV(cv=3), binwidth=binwidth)
    nglm.fit(train_idx=train_trial_ids[i])
    
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

glm_score_save_path = save_data_path(figure='figure9')
np.save(glm_score_save_path.joinpath('glm_scores.npy'), scores)

num_trials=350
n_test = int(num_trials*0.2)
n_train = int((num_trials-n_test)*0.8)
n_val = num_trials - n_train - n_test

test_idx = np.random.choice(np.arange(num_trials), size=n_test, replace=False)
test_bool = np.zeros(num_trials).astype(bool)
test_bool[test_idx] = True

train_idx = np.random.choice(np.arange(num_trials)[~test_bool], size=n_train, replace=False)
train_bool = np.zeros(num_trials).astype(bool)
train_bool[train_idx] = True

val_bool = np.zeros(num_trials).astype(bool)
val_bool[~np.logical_or(test_bool, train_bool)] = True

train_idx = np.arange(num_trials)[train_bool]
val_idx = np.arange(num_trials)[val_bool]
test_idx = np.arange(num_trials)[test_bool]

rng = np.random.default_rng(seed=0b01101001 + 0b01100010 + 0b01101100)

fdb_rt_vals = np.linspace(0.1, 0.7, num=10)
fdb_rt_probs = np.array([0.15970962, 0.50635209, 0.18693285, 0.0707804, 0.02540835,
                     0.01633394, 0.00907441, 0.00725953, 0.00544465, 0.01270417])

stim_rt_vals = np.linspace(0.25, 0.4, num=10)
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

for i, eid in notebook.tqdm(enumerate(mtnn_eids)):
    print('processing session {}'.format(eid))
    
    trialsdf = trialsdf_list[i]
    nglm = fit_glm_lists[i]
    clus = cluster_list[i]
    weights = nglm.combine_weights()

    wheeltraces = trialsdf.wheel_velocity.to_list()
    firstMovement_times = np.ones(num_trials) * 0.5
    stimtimes = rng.choice(stim_rt_vals, size=num_trials, p=stim_rt_probs) \
        + normal(size=num_trials) * 0.05
    fdbktimes = rng.choice(fdb_rt_vals, size=num_trials, p=fdb_rt_probs) \
        + firstMovement_times + normal(size=num_trials) * 0.05
    if prior_list[i].shape[0] < num_trials:
        priors = np.concatenate((prior_list[i], prior_list[i][:num_trials-prior_list[i].shape[0]]))
    else:
        priors = prior_list[i][:num_trials]
    prev_priors = np.pad(priors, (1,0), constant_values=0)[:-1]
    contrasts = rng.choice(contrastvals, replace=True, size=num_trials)
    feedbacktypes = rng.choice([-1, 1], size=num_trials, p=[0.1, 0.9])
    wheelmoves = rng.choice(np.arange(len(wheeltraces)), size=num_trials)
    
    session_simulated_spkidx_list = []
    session_simulated_feature_list = []
    
    print(f'total number of units: {len(clus)}')
    
    for j, clu in notebook.tqdm(enumerate(clus)):
        
        scales = np.random.uniform(1, 2, size=6)
        scales_dict[(eid,clu)] = scales
        
        stimkernL = weights['stimonL'].loc[clu].to_numpy() * (1/binwidth) * scales[0]
        stimkernR = weights['stimonR'].loc[clu].to_numpy() * (1/binwidth) * scales[1]
        fdbkkern1 = weights['correct'].loc[clu].to_numpy() * (1/binwidth) * scales[2]
        fdbkkern2 = weights['incorrect'].loc[clu].to_numpy() * (1/binwidth) * scales[3]
        wheelkern = weights['wheel'].loc[clu].to_numpy() * (1/binwidth) * scales[4]
        fmovekern = weights['fmove'].loc[clu].to_numpy() * (1/binwidth) * scales[5]
    
        ret = simulate_cell((stimkernL,stimkernR), (fdbkkern1,fdbkkern2), 
                            fmovekern, wheelkern, 
                            wheeltraces,
                            stimtimes, fdbktimes, feedbacktypes, firstMovement_times,
                            priors, prev_priors, contrasts,
                            wheelmoves, pgain=2.0, gain=8.0, 
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
                        estimator=RidgeCV(cv=3), binwidth=binwidth)
        nglm.fit(train_idx=train_idx)
        
#         pred = GLMPredictor(nglm, adj_spkt, sess_clu, new_trialsdf, np.arange(num_trials))
#         for j, unit in enumerate(np.unique(sess_clu)):
#             ax = pred.psth_summary('firstMovement_times', unit, t_before=0.5, t_after=1.0)
#             plt.show()
        
        score = nglm.score()
        simulated_glm_scores.append(score)

        sfs = mut.SequentialSelector(nglm, train=train_idx, test=test_idx, direction='backward', 
                                     n_features_to_select=len(nglm.design.covar)-1)
        sfs.fit(progress=True)
        sfs.all_scores.loc[unit_id] = score.loc[unit_id] - sfs.all_scores.loc[unit_id]
        simulated_glm_leave_one_out.append(sfs.all_scores)
#         plt.bar(glm_covs.keys(), sfs.all_scores.loc[unit_id].to_numpy())
#         plt.show()

        unit_id += 1
        
#         nbins = int(1.5/binwidth)
#         raster = np.zeros((len(new_trialsdf), nbins))
#         for trial in range(len(new_trialsdf)):
#             for n in range(nbins):
#                 idx = np.logical_and(adj_spkt>=1.5*trial+binwidth*n, adj_spkt<1.5*trial+binwidth*(n+1))
#                 raster[trial,n] = idx.astype(int).sum() / binwidth
                
#         plt.figure(figsize=(8,2))
#         plt.plot(raster.mean(0), color='k')
#         plt.show()
                
        session_simulated_spkidx_list.append(raster[None])
    simulated_output_list.append(np.concatenate(session_simulated_spkidx_list, axis=0))
    simulated_feature_list.append(np.concatenate(session_simulated_feature_list, axis=0))
simulated_glm_leave_one_out = pd.concat(simulated_glm_leave_one_out)
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
    
save_path = save_data_path(figure='figure10').joinpath('simulated_data')
save_path.mkdir(exist_ok=True, parents=True)

save_path_train = save_data_path(figure='figure10').joinpath('simulated_data/train')
save_path_train.mkdir(exist_ok=True, parents=True)

save_path_val = save_data_path(figure='figure10').joinpath('simulated_data/validation')
save_path_val.mkdir(exist_ok=True, parents=True)

save_path_test = save_data_path(figure='figure10').joinpath('simulated_data/test')
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
np.save(save_path_val.joinpath('feature.npy'), np.concatenate(val_output))
np.save(save_path_test.joinpath('feature.npy'), np.concatenate(test_output))

np.save(save_path.joinpath('glm_scores.npy'), simulated_glm_scores)
np.save(save_path.joinpath('glm_leave_one_out.npy'), simulated_glm_leave_one_out)

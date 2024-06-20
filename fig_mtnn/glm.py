"""
Code for GLM
@author Hyun Dong Lee
"""
import numpy as np
import pandas as pd
import fig_mtnn.modeling.design_matrix as dm
import fig_mtnn.modeling.utils as mut

import matplotlib.pyplot as plt
from fig_mtnn.modeling import linear
from fig_mtnn.modeling import poisson
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

    #print('Condition of design matrix:', np.linalg.cond(design.dm))
    return design

bases_full_mtnn_cov = {
    'stim': mut.raised_cosine(0.4, 5, tmp_binf),
    'goCue': mut.raised_cosine(0.4, 5, tmp_binf),
    'choice': mut.raised_cosine(0.4, 5, tmp_binf),
    'feedback': mut.raised_cosine(0.4, 5, tmp_binf),
    'wheel': mut.raised_cosine(0.3, 3, tmp_binf),
    'dlc': mut.raised_cosine(0.3, 3, tmp_binf),
    'fmove': mut.raised_cosine(0.2, 3, tmp_binf),
}

def generate_design_full_mtnn_cov(trialsdf, prior, t_before, bases, prior_last=None,
                    iti_prior=[-0.4, -0.1], fmove_offset=-0.4, wheel_offset=-0.4,
                    contnorm=5., duration=None, binwidth=0.05, reduce_wheel_dim=True):
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
                'paw_speed': 'continuous',
                'nose_speed': 'continuous',
                'pupil_diameter': 'continuous',
                'left_me': 'continuous',
                'lick': 'continuous',
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

    design = dm.DesignMatrix(trialsdf, vartypes, duration=duration, binwidth=binwidth)
#     stepbounds = [design.binf(t_before + iti_prior[0]), design.binf(t_before + iti_prior[1])]
#     print(stepbounds)

    design.add_covariate_timing('stimonL', 'stimOn_times', bases_full_mtnn_cov['stim'],
                                cond=lambda tr: np.isfinite(tr.contrastLeft),
                                deltaval='adj_contrastL',
                                desc='Kernel conditioned on L stimulus onset')
    design.add_covariate_timing('stimonR', 'stimOn_times', bases_full_mtnn_cov['stim'],
                                cond=lambda tr: np.isfinite(tr.contrastRight),
                                deltaval='adj_contrastR',
                                desc='Kernel conditioned on R stimulus onset')
    
    design.add_covariate_timing('go', 'goCue_times', bases_full_mtnn_cov['goCue'],
                                cond=lambda tr: True,
                                desc='Kernel conditioned on go cue timing')
    
    design.add_covariate_timing('ccw', 'response_times', bases_full_mtnn_cov['choice'],
                                cond=lambda tr: tr.choice == -1,
                                desc='Kernel conditioned on ccw choice')
    design.add_covariate_timing('cw', 'response_times', bases_full_mtnn_cov['choice'],
                                cond=lambda tr: tr.choice == 1,
                                desc='Kernel conditioned on cw choice')
    
    design.add_covariate_timing('correct', 'feedback_times', bases_full_mtnn_cov['feedback'],
                                cond=lambda tr: tr.feedbackType == 1,
                                desc='Kernel conditioned on correct feedback')
    design.add_covariate_timing('incorrect', 'feedback_times', bases_full_mtnn_cov['feedback'],
                                cond=lambda tr: tr.feedbackType == -1,
                                desc='Kernel conditioned on incorrect feedback')
    
    design.add_covariate_timing('fmove', 'firstMovement_times', bases_full_mtnn_cov['fmove'],
                                offset=fmove_offset,
                                desc='Lead up to first movement')
    
    design.add_covariate_raw('pLeft', stepfunc_prestim,
                             desc='Step function on prior estimate')
    design.add_covariate_raw('pLeft_tr', stepfunc_poststim,
                             desc='Step function on post-stimulus prior')

    design.add_covariate('paw', trialsdf['paw_speed'], bases_full_mtnn_cov['dlc'])
    design.add_covariate('nose', trialsdf['nose_speed'], bases_full_mtnn_cov['dlc'])
    design.add_covariate('pupil', trialsdf['pupil_diameter'], bases_full_mtnn_cov['dlc'])
    design.add_covariate('left_me', trialsdf['left_me'], bases_full_mtnn_cov['dlc'])
    design.add_covariate('lick', trialsdf['lick'], bases_full_mtnn_cov['dlc'])
    design.add_covariate('wheel', trialsdf['wheel_velocity'], bases_full_mtnn_cov['wheel'], wheel_offset)
    
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

    #print('Condition of design matrix:', np.linalg.cond(design.dm))
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


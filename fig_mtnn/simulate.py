"""
Code to create simulated data
@author Hyun Dong Lee
"""

import numpy as np
import pandas as pd
from fig_mtnn.modeling.design_matrix import convbasis
from scipy.interpolate import interp1d
from numpy.random import uniform, normal

binwidth = 0.05

rng = np.random.default_rng(seed=123456)

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
            spikecounts = rng.poisson(ratevals)
        else:
            ratevals = (kernsum + gain) * binwidth
            contspikecounts = rng.normal(
                loc=ratevals, scale=gain * binwidth * 7e-1)
            spikecounts = np.round(contspikecounts).astype(int)
        if ret_raw:
            trialcont.append(contspikecounts)
        spike_times = []

        noisevals = rng.uniform(low=0, high=binwidth/2,
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
    trialends = np.pad(trialends, (1, 0), constant_values=0)
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

    contrastLeft[np.where(contrasts > 0)] = contrasts[np.where(contrasts > 0)]
    contrastRight[np.where(contrasts < 0)] = contrasts[np.where(contrasts < 0)]

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
    feature = np.zeros((len(trialsdf), trial_len, nfeats + 1))
    for i in range(len(trialsdf)):
        stimon_bin = (trialsdf.loc[i]['stimOn_times'] - trialsdf.loc[i]['trial_start']) / binwidth
        stimon_bin = np.floor(stimon_bin).astype(int)

        feedback_bin = (trialsdf.loc[i]['feedback_times'] - trialsdf.loc[i]['trial_start']) / binwidth
        feedback_bin = np.floor(feedback_bin).astype(int)

        fmv_bin = (trialsdf.loc[i]['firstMovement_times'] - trialsdf.loc[i]['trial_start']) / binwidth
        fmv_bin = np.floor(fmv_bin).astype(int)

        if np.isnan(trialsdf.loc[i]['contrastRight']):
            feature[i, stimon_bin, 1] = np.abs(trialsdf.loc[i]['contrastLeft'])
        else:
            feature[i, stimon_bin, 2] = np.abs(trialsdf.loc[i]['contrastRight'])

        if trialsdf.loc[i]['feedbackType'] == -1:
            feature[i, feedback_bin, 3] = 1
        else:
            feature[i, feedback_bin, 4] = 1

        feature[i, fmv_bin, 5] = 1

        feature[i, :, 6] = trialsdf.loc[i]['prior']
        feature[i, :, 7] = trialsdf.loc[i]['prior_last']
        feature[i, :, 8] = trialsdf.loc[i]['wheel_velocity']

    return feature

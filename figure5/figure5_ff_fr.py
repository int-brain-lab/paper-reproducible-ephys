import numpy as np
import pandas as pd
import seaborn as sns
from os.path import join
import matplotlib.pyplot as plt
import brainbox.io.one as bbone
from reproducible_ephys_functions import query, labs, eid_list, combine_regions, get_insertions
from reproducible_ephys_paths import FIG_PATH
from one.api import ONE
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.tri as mtri

import traceback

from pylab import *
import scipy.io as sio
from brainbox.singlecell import calculate_peths
from brainbox.plot import peri_event_time_histogram, firing_rate
from brainbox.metrics.single_units import spike_sorting_metrics, quick_unit_metrics
import pickle

from ibllib.atlas import atlas, AllenAtlas
from ibllib.pipes import histology
from iblutil.util import Bunch
from iblutil.numerical import ismember

from brainbox.io.one import SpikeSortingLoader
# from brainbox.task.trials import find_trial_ids
from scipy.stats import wilcoxon#, ranksums, ttest_ind, ttest_rel
from brainbox.task._statsmodels import multipletests
import time


one = ONE()
ba = AllenAtlas()

insertions = get_insertions(level=2, as_dataframe=False)

TRIAL_ATTRIBUTES = ['choice', 'contrastLeft', 'contrastRight', 'feedback_times', 'feedbackType', 'firstMovement_times',
                    'stimOn_times']

REGION = 'LP'
binSzFRPeri = 0.04  # Bin size for calculating perievent FR when using sliding window
binSzFFPeri = 0.1  # Bin size for calculating perievent FF when using sliding window
n_slideFR = 2  # Number of slides per bin for FR calculation when using sliding window
n_slideFF = 5  # Number of slides per bin for FF calculation when using sliding window
pre_time = 0.4
post_time = 0.8
CapPostTime = 0.4  # Time point at which to cap the avg FR/FF post-event; even though over time we have 0.8 s post-event, we can cap the analysis of avg post-event to 0.4 s
Caus = 1

NoGoodClustersInRegion = []



for ins in insertions:
    pid = ins['probe_insertion']
    eid = ins['session']['id']
    probe = ins['probe_name']
    DateInfo = ins['session']['start_time']
    subj = ins['session']['subject']

    # Load in all the data that we need
    try:
        # Should this be alfio

        sl = SpikeSortingLoader(pid, one=one, atlas=ba)
        spikes, clusters, channels = sl.load_spike_sorting(dataset_types=['clusters.amps', 'clusters.peakToTrough'])
        clusters = sl.merge_clusters(spikes, clusters, channels)
        trials = one.load_object(eid, 'trials')

        clusters['rep_site_acronym'] = combine_regions(clusters['acronym'])
#
        clusterIDs = clusters['cluster_id'][
            np.bitwise_and(clusters['rep_site_acronym'] == REGION, clusters['label'] == 1)]


    #if len(clusterIDs) == 0:  # when no good clusters are found in the brain region
    #    NoGoodClustersInRegion.append(subj)
    #    continue

        start = time.time()
        nanStim = np.isnan(trials['stimOn_times'])
        nanMovement = np.isnan(trials['firstMovement_times'])
        nanFeedback = np.isnan(trials['feedback_times'])
        nanStimMovement = np.bitwise_or(nanStim, nanMovement)

        print(f'stimOn_nans: {np.sum(nanStim)}')
        print(f'movement_nans: {np.sum(nanMovement)}')
        print(f'feedback_nans: {np.sum(nanFeedback)}')
        print(f'stim and movement_nans: {np.sum(nanStimMovement)}')

        trials_id = np.arange(trials['feedbackType'].shape[0])

        # Make a dict of all trials permutations that we will use later for analysis
        # fb - feedback, either Corr (correct) or Incorr (incorrect)
        # st - stimulus side, either L (left) or R (right)
        # ch - choice, either L (leftward wheel motion (1)) or R (rightward wheel motion (-1))
        # number - stimulus contrast percentage
        trials_outcome = {}
        # stim: Left + Right, choice: Left + Right, feedback: Corr + Incorr, contrasts: 0
        trials_outcome['0'] = np.bitwise_or(trials['contrastLeft'] == 0, trials['contrastRight'] == 0)
        # stim: Left + Right, choice: Left + Right, feedback: Corr, contrasts: all
        trials_outcome['fbCorr'] = trials['feedbackType'] == 1
        # stim: Left + Right, choice: Left + Right, feedback: Corr, contrasts: all
        trials_outcome['fbIncorr'] = trials['feedbackType'] == -1
        # stim: Left + Right, choice: Left + Right, feedback: Corr, contrasts: > 0
        trials_outcome['fbCorr_gt0'] = np.bitwise_and(trials_outcome['fbCorr'], ~trials_outcome['0'])
        # stim: Left + Right, choice: Left + Right, feedback: Incorr, contrasts: > 0
        trials_outcome['fbIncorr_gt0'] = np.bitwise_and(trials_outcome['fbIncorr'], ~trials_outcome['0'])
        # stim: Left, choice: Left + Right, feedback: Corr + Incorr, contrasts: > 0
        trials_outcome['stL'] = trials['contrastLeft'] > 0
        # stim: Right, choice: Left + Right, feedback: Corr + Incorr, contrasts: > 0
        trials_outcome['stR'] = trials['contrastRight'] > 0
        # stim: Left + Right, choice: Left, feedback: Corr + Incorr, contrasts: all
        trials_outcome['chL'] = trials['choice'] == 1
        # stim: Left + Right, choice: Right, feedback: Corr + Incorr, contrasts: all
        trials_outcome['chR'] = trials['choice'] == -1

        # stim: Left, choice: Left, feedback: Corr + Incorr, contrasts: > 0
        trials_outcome['stL_chL'] = np.bitwise_and(trials_outcome['stL'], trials_outcome['chL'])
        # stim: Left, choice: Right, feedback: Corr + Incorr, contrasts: > 0
        trials_outcome['stL_chR'] = np.bitwise_and(trials_outcome['stL'], trials_outcome['chR'])
        # stim: Right, choice: Left, feedback: Corr + Incorr, contrasts: > 0
        trials_outcome['stR_chL'] = np.bitwise_and(trials_outcome['stR'], trials_outcome['chL'])
        # stim: Right, choice: Right, feedback: Corr + Incorr, contrasts: > 0
        trials_outcome['stR_chR'] = np.bitwise_and(trials_outcome['stR'], trials_outcome['chR'])

        # stim: Left, choice: Left + Right, feedback: Corr, contrasts: > 0 (it hink this should be >= 0)
        trials_outcome['stL_fbCorr'] = np.bitwise_and(trials_outcome['stL'], trials_outcome['fbCorr'])
        # stim: Left, choice: Left + Right, feedback: Incorr, contrasts: > 0
        trials_outcome['stL_fbIncorr'] = np.bitwise_and(trials_outcome['stL'], trials_outcome['fbIncorr'])
        # stim: Right, choice: Left + Right, feedback: Corr, contrasts: > 0
        trials_outcome['stR_fbCorr'] = np.bitwise_and(trials_outcome['stR'], trials_outcome['fbCorr'])
        # stim: Right, choice: Left + Right, feedback: Incorr, contrasts: > 0
        trials_outcome['stR_fbIncorr'] = np.bitwise_and(trials_outcome['stR'], trials_outcome['fbIncorr'])

        # sanity check - stim: left, feedback: correct should be the same as stim: left, choice: left
        np.testing.assert_equal(trials_outcome['stL_fbCorr'], trials_outcome['stL_chL'])
        # sanity check - stim: right, feedback: correct should be the same as stim: right, choice: right
        np.testing.assert_equal(trials_outcome['stR_fbCorr'], trials_outcome['stR_chR'])

        # stim: Left, choice: Left + Right, feedback: Corr + Incorr, contrasts: 100
        trials_outcome['stL_100'] = trials['contrastLeft'] == 1
        # stim: Right, choice: Left + Right, feedback: Corr + Incorr, contrasts: 100
        trials_outcome['stR_100'] = trials['contrastRight'] == 1
        # stim: Left, choice: Left, feedback: Corr + Incorr, contrasts: 100
        trials_outcome['stL_100_chL'] = np.bitwise_and(trials_outcome['stL_100'], trials_outcome['chL'])
        # stim: Left, choice: Right, feedback: Corr + Incorr, contrasts: 100
        trials_outcome['stL_100_chR'] = np.bitwise_and(trials_outcome['stL_100'], trials_outcome['chR'])
        # stim: Right, choice: Left, feedback: Corr + Incorr, contrasts: 100
        trials_outcome['stR_100_chL'] = np.bitwise_and(trials_outcome['stR_100'], trials_outcome['chL'])
        # stim: Right, choice: Right, feedback: Corr + Incorr, contrasts: 100
        trials_outcome['stR_100_chR'] = np.bitwise_and(trials_outcome['stR_100'], trials_outcome['chR'])

        # Get all the events of interest
        events = {}
        # STIM EVENTS
        # remove trials that have nans in stimOn_times
        events['eventStim_stL'] = trials['stimOn_times'][np.bitwise_and(trials_outcome['stL'], ~nanStim)]
        events['eventStim_stR'] = trials['stimOn_times'][np.bitwise_and(trials_outcome['stR'], ~nanStim)]
        events['eventStim_0'] = trials['stimOn_times'][np.bitwise_and(trials_outcome['0'], ~nanStim)]
        events['eventStim_stL100'] = trials['stimOn_times'][np.bitwise_and(trials_outcome['stL_100'], ~nanStim)]
        events['eventStim_stR100'] = trials['stimOn_times'][np.bitwise_and(trials_outcome['stR_100'], ~nanStim)]
        events['eventStim_stLfbCorr'] = trials['stimOn_times'][np.bitwise_and(trials_outcome['stL_fbCorr'], ~nanStim)]
        events['eventStim_stLfbIncorr'] = trials['stimOn_times'][np.bitwise_and(trials_outcome['stL_fbIncorr'], ~nanStim)]
        events['eventStim_stRfbCorr'] = trials['stimOn_times'][np.bitwise_and(trials_outcome['stR_fbCorr'], ~nanStim)]
        events['eventStim_stRfbIncorr'] =  trials['stimOn_times'][np.bitwise_and(trials_outcome['stR_fbIncorr'], ~nanStim)]
        # remove trials that have nans in stimOn_times or firstMovement_times
        events['eventStim_All'] = trials['stimOn_times'][~nanStimMovement]
        events['eventStim_fbCorr'] = trials['stimOn_times'][np.bitwise_and(trials_outcome['fbCorr_gt0'], ~nanStimMovement)]
        events['eventStim_fbIncorr'] = trials['stimOn_times'][np.bitwise_and(trials_outcome['fbIncorr_gt0'], ~nanStimMovement)]
        events['eventStim_stLchL'] = trials['stimOn_times'][np.bitwise_and(trials_outcome['stL_chL'], ~nanStimMovement)]
        events['eventStim_stLchR'] = trials['stimOn_times'][np.bitwise_and(trials_outcome['stL_chR'], ~nanStimMovement)]
        events['eventStim_stRchL'] = trials['stimOn_times'][np.bitwise_and(trials_outcome['stR_chL'], ~nanStimMovement)]
        events['eventStim_stRchR'] = trials['stimOn_times'][np.bitwise_and(trials_outcome['stR_chR'], ~nanStimMovement)]
        events['eventStim_stL100chL'] = trials['stimOn_times'][np.bitwise_and(trials_outcome['stL_100_chL'], ~nanStimMovement)]
        events['eventStim_stL100chR'] = trials['stimOn_times'][np.bitwise_and(trials_outcome['stL_100_chR'], ~nanStimMovement)]
        events['eventStim_stR100chL'] = trials['stimOn_times'][np.bitwise_and(trials_outcome['stR_100_chL'], ~nanStimMovement)]
        events['eventStim_stR100chR'] = trials['stimOn_times'][np.bitwise_and(trials_outcome['stR_100_chR'], ~nanStimMovement)]

        # MOVE EVENTS
        # remove trials that have nans in firstMovement_times
        events['eventMove_chL'] = trials['firstMovement_times'][np.bitwise_and(trials_outcome['chL'], ~nanMovement)]
        events['eventMove_chR'] = trials['firstMovement_times'][np.bitwise_and(trials_outcome['chR'], ~nanMovement)]
        # remove trials that have nans in stimOn_times or firstMovement_times
        events['eventMove_All'] = trials['firstMovement_times'][~nanStimMovement]
        events['eventMove_fbCorr'] = trials['firstMovement_times'][np.bitwise_and(trials_outcome['fbCorr_gt0'], ~nanStimMovement)]
        events['eventMove_fbIncorr'] = trials['firstMovement_times'][np.bitwise_and(trials_outcome['fbIncorr_gt0'], ~nanStimMovement)]
        events['eventMove_stLfbCorr'] = trials['firstMovement_times'][np.bitwise_and(trials_outcome['stL_fbCorr'], ~nanStimMovement)]
        events['eventMove_stLfbIncorr'] = trials['firstMovement_times'][np.bitwise_and(trials_outcome['stL_fbIncorr'], ~nanStimMovement)]
        events['eventMove_stRfbCorr'] = trials['firstMovement_times'][np.bitwise_and(trials_outcome['stR_fbCorr'], ~nanStimMovement)]
        events['eventMove_stRfbIncorr'] =  trials['firstMovement_times'][np.bitwise_and(trials_outcome['stR_fbIncorr'], ~nanStimMovement)]
        events['eventMove_stLchL'] = trials['firstMovement_times'][np.bitwise_and(trials_outcome['stL_chL'], ~nanStimMovement)]
        events['eventMove_stLchR'] = trials['firstMovement_times'][np.bitwise_and(trials_outcome['stL_chR'], ~nanStimMovement)]
        events['eventMove_stRchL'] = trials['firstMovement_times'][np.bitwise_and(trials_outcome['stR_chL'], ~nanStimMovement)]
        events['eventMove_stRchR'] = trials['firstMovement_times'][np.bitwise_and(trials_outcome['stR_chR'], ~nanStimMovement)]
        events['eventMove_stL100chL'] = trials['firstMovement_times'][np.bitwise_and(trials_outcome['stL_100_chL'], ~nanStimMovement)]
        events['eventMove_stL100chR'] = trials['firstMovement_times'][np.bitwise_and(trials_outcome['stL_100_chR'], ~nanStimMovement)]
        events['eventMove_stR100chL'] = trials['firstMovement_times'][np.bitwise_and(trials_outcome['stR_100_chL'], ~nanStimMovement)]
        events['eventMove_stR100chR'] = trials['firstMovement_times'][np.bitwise_and(trials_outcome['stR_100_chR'], ~nanStimMovement)]

        # FEEDBACK EVENTS
        # remove trials that have nans in feedback_times
        events['eventFdbck_fbCorr'] = trials['feedback_times'][np.bitwise_and(trials_outcome['fbCorr'], ~nanFeedback)]
        events['eventFdbck_fbIncorr'] = trials['feedback_times'][np.bitwise_and(trials_outcome['fbIncorr'], ~nanFeedback)]

        # REACTION TIMES
        rxn_times = trials['firstMovement_times'] - trials['stimOn_times']
        events['rxnTimes_All'] = rxn_times[~nanStimMovement]
        events['rxnTimes_fbCorr'] = rxn_times[np.bitwise_and(trials_outcome['fbCorr_gt0'], ~nanStimMovement)]
        events['rxnTimes_stLfbCorr'] = rxn_times[np.bitwise_and(trials_outcome['stL_fbCorr'], ~nanStimMovement)]
        events['rxnTimes_stRfbCorr'] = rxn_times[np.bitwise_and(trials_outcome['stR_fbCorr'], ~nanStimMovement)]

        # RESTRUCTURED TIMES
        trials['restr_stimOn'] = np.copy(trials['stimOn_times'])
        trials['restr_stimOn'][rxn_times > 0.2] = trials['firstMovement_times'][rxn_times > 0.2] - 0.2
        events['eventStimRestr_All'] = trials['restr_stimOn'][np.bitwise_and(rxn_times > 0.05, ~nanStimMovement)]
        events['eventStimRestr_fbCorr'] = trials['restr_stimOn'][np.bitwise_and(np.bitwise_and(trials_outcome['fbCorr_gt0'],
                                                                                               rxn_times > 0.05), ~nanStimMovement)]
        events['eventMoveRestr_All'] = trials['firstMovement_times'][np.bitwise_and(rxn_times > 0.05, ~nanStimMovement)]
        events['eventMoveRestr_fbCorr'] = trials['firstMovement_times'][np.bitwise_and(np.bitwise_and(trials_outcome['fbCorr_gt0'],
                                                                                               rxn_times > 0.05), ~nanStimMovement)]
        events['rxnTimesRestr_All'] = events['eventMoveRestr_All'] - events['eventStimRestr_All']
        events['rxnTimesRestr_fbCorr'] = events['eventMoveRestr_fbCorr'] - events['eventStimRestr_fbCorr']

        print(time.time() - start)

        # Find specific task Event times:
        # (Could potentially replace the following with the find_trial_IDs function)
        # event == 'Stim'
        start = time.time()
        timesStimOn_orig = one.load_dataset(eid,
                                            '_ibl_trials.stimOn_times.npy')  # one.load(eid, dataset_types=['trials.stimOn_times'])[0]

        contrast_L_orig = one.load_dataset(eid, '_ibl_trials.contrastLeft.npy')
        contrast_R_orig = one.load_dataset(eid, '_ibl_trials.contrastRight.npy')
        contrast_L, contrast_R = contrast_L_orig[~np.isnan(timesStimOn_orig)], contrast_R_orig[~np.isnan(timesStimOn_orig)]
        timesStimOn = timesStimOn_orig[~np.isnan(timesStimOn_orig)]
        event_times_left = timesStimOn[contrast_L > 0]
        event_times_right = timesStimOn[contrast_R > 0]
        event_times_0 = timesStimOn[
            np.logical_or(contrast_R == 0, contrast_L == 0)]  # 'or' is used because the other contrast is always nan
        event_times_left100 = timesStimOn[contrast_L == 1]
        event_times_right100 = timesStimOn[contrast_R == 1]

        # event == 'Move'
        times1stMove_orig = one.load_dataset(eid,
                                             '_ibl_trials.firstMovement_times.npy')  # one.load(eid, dataset_types=['trials.firstMovement_times'])[0]
        choice_orig = one.load_dataset(eid, '_ibl_trials.choice.npy')  # one.load(eid, dataset_types=['trials.choice'])[0]
        #if (~np.isnan(
        #        times1stMove_orig)).sum() < 300:  # don't analyze if mouse made less than 300 choices/movement (even though >400 trials were done)
        #    continue
        choice = choice_orig[~np.isnan(times1stMove_orig)]
        times1stMove = times1stMove_orig[~np.isnan(times1stMove_orig)]
        event_times_Rchoice = times1stMove[choice == -1]  # -1 means mouse reports stim on Right, so counter-clockwise wheel turn
        event_times_Lchoice = times1stMove[choice == 1]  # 1 means mouse reports stim on Left, so clockwise wheel turn
        StimWithChoice = np.logical_and(~np.isnan(timesStimOn_orig),
                                        ~np.isnan(times1stMove_orig))  # all the indices where there was both a stim and a choice time
        # MoveandRstim = np.logical_and(contrast_R_orig>0, StimWithChoice) #events when stim was R (contrast>0) and there was a choice ultimately
        MoveandR100stim = np.logical_and(contrast_R_orig == 1,
                                         StimWithChoice)  # events when stim was R (contrast>0) and there was a choice ultimately
        MoveandL100stim = np.logical_and(contrast_L_orig == 1, StimWithChoice)
        eventMove_stR100chR = times1stMove_orig[
            np.logical_and(MoveandR100stim, choice_orig == -1)]  # times of movement when choice was R and stim was R (contrast>0)
        eventMove_stR100chL = times1stMove_orig[np.logical_and(MoveandR100stim,
                                                               choice_orig == 1)]  # times of movement when choice was L and stim was R (contrast>0), so incorrect
        eventMove_stL100chR = times1stMove_orig[np.logical_and(MoveandL100stim, choice_orig == -1)]
        eventMove_stL100chL = times1stMove_orig[np.logical_and(MoveandL100stim, choice_orig == 1)]
        eventStim_stR100chR = timesStimOn_orig[
            np.logical_and(MoveandR100stim, choice_orig == -1)]  # times of stim onset when choice was R and stim was R (contrast>0)
        eventStim_stR100chL = timesStimOn_orig[np.logical_and(MoveandR100stim,
                                                              choice_orig == 1)]  # times of stim onset when choice was L and stim was R (contrast>0), so incorrect
        eventStim_stL100chR = timesStimOn_orig[np.logical_and(MoveandL100stim, choice_orig == -1)]
        eventStim_stL100chL = timesStimOn_orig[np.logical_and(MoveandL100stim, choice_orig == 1)]
        # Aligned to movement, but for any nonzero contrast:
        MoveandRstim = np.logical_and(contrast_R_orig > 0,
                                      StimWithChoice)  # events when stim was R (contrast>0) and there was a choice ultimately
        MoveandLstim = np.logical_and(contrast_L_orig > 0, StimWithChoice)
        eventMove_stRchR = times1stMove_orig[
            np.logical_and(MoveandRstim, choice_orig == -1)]  # times of movement when choice was R and stim was R (contrast>0)
        eventMove_stRchL = times1stMove_orig[np.logical_and(MoveandRstim,
                                                            choice_orig == 1)]  # times of movement when choice was L and stim was R (contrast>0), so incorrect
        eventMove_stLchR = times1stMove_orig[np.logical_and(MoveandLstim, choice_orig == -1)]
        eventMove_stLchL = times1stMove_orig[np.logical_and(MoveandLstim, choice_orig == 1)]


        timesFeedback_orig = one.load_dataset(eid, '_ibl_trials.feedback_times.npy')
        FeedbackType_orig = one.load_dataset(eid, '_ibl_trials.feedbackType.npy')
        FeedbackType = FeedbackType_orig[~np.isnan(timesStimOn_orig)] #Or should it be: FeedbackType_orig[~np.isnan(timesFeedback_orig)]? Sometimes a mismatch occurs
        #TO DO: Need to understand the discripency between event_CorrR and the following:
        #event_stRchR = times1stMove_orig[np.logical_and(MoveandRstim, choice_orig == -1)] #times of movement when choice was R and stim was R (contrast>0)
        #(i.e., mistmatch between shape(event_stLchL), e.g. =149, and shape(event_CorrL), e.g. =150.
        #How can the fdback be 1 if the mouse didn't move?; depending on result, may need to change the calculations in brainbox -> trials.py)
        event_CorrR = timesStimOn[np.logical_and(FeedbackType==1, contrast_R>0)] #Any non-zero R contrast that's correct
        event_CorrL = timesStimOn[np.logical_and(FeedbackType==1, contrast_L>0)]
        event_IncorrR = timesStimOn[np.logical_and(FeedbackType==-1, contrast_R>0)] #Any non-zero R contrast that's correct
        event_IncorrL = timesStimOn[np.logical_and(FeedbackType==-1, contrast_L>0)]

        timesFeedback = timesFeedback_orig[~np.isnan(timesFeedback_orig)]
        FeedbackType2 = FeedbackType_orig[~np.isnan(timesFeedback_orig)]
        event_FdbckCorr = timesFeedback[FeedbackType2 == 1] #Times when reward was given
        event_FdbckIncorr = timesFeedback[FeedbackType2 == -1] #Times when noise burst was given


        #Reaction time (i.e., time from stim On until 1st movement):
        times1stMove_orig2 = times1stMove_orig[~np.isnan(timesStimOn_orig)]
        RxnTimes = times1stMove_orig2-timesStimOn #include the _orig so that any NaNs stay nan
        #TrialsToExclude = np.where(np.isnan(RxnTimes)) #either NaN movement or stim
        RxnTimes_fullTrial = RxnTimes[~np.isnan(RxnTimes)]
        times1stMove_fullTrial = times1stMove_orig2[~np.isnan(RxnTimes)]
        timesStimOn_fullTrial = timesStimOn[~np.isnan(RxnTimes)]

        #Same as above, but restrict to only correct trials & constrast>0:
        RxnTimes_CorrTrialWithNan = RxnTimes[np.logical_or(np.logical_and(FeedbackType==1, contrast_R>0),
                                                    np.logical_and(FeedbackType==1, contrast_L>0))]
        RxnTimes_CorrTrial = RxnTimes_CorrTrialWithNan[~np.isnan(RxnTimes_CorrTrialWithNan)]
        times1stMove_CorrTrial = times1stMove_orig2[np.logical_or(np.logical_and(FeedbackType==1, contrast_R>0),
                                                                 np.logical_and(FeedbackType==1, contrast_L>0))]
        times1stMove_CorrTrial = times1stMove_CorrTrial[~np.isnan(RxnTimes_CorrTrialWithNan)]
        timesStimOn_CorrTrial = timesStimOn[np.logical_or(np.logical_and(FeedbackType==1, contrast_R>0),
                                                               np.logical_and(FeedbackType==1, contrast_L>0))]
        timesStimOn_CorrTrial = timesStimOn_CorrTrial[~np.isnan(RxnTimes_CorrTrialWithNan)]

        # In the case below, restrict trials to:
        # (1) Up to only 200 ms pre-movement (so not the long waiting periods):
        timesStart = timesStimOn
        timesStart[RxnTimes > 0.2] = times1stMove_orig2[RxnTimes > 0.2] - 0.2
        # (2) Trials with >50 ms between stim and movement:
        timesStart_restrTrials1 = timesStart[RxnTimes > 0.05]
        times1stMove_restrTrials1 = times1stMove_orig2[RxnTimes > 0.05]
        RxnTimes_restrTrials1 = times1stMove_restrTrials1 - timesStart_restrTrials1
        # (3) trials that were correct choices and constrast>0 (from (1), then apply (2) again):
        timesStart_restrTrialsTemp = timesStart[np.logical_or(np.logical_and(FeedbackType == 1, contrast_R > 0),
                                                              np.logical_and(FeedbackType == 1, contrast_L > 0))]
        times1stMove_restrTrialsTemp = times1stMove_orig2[np.logical_or(np.logical_and(FeedbackType == 1, contrast_R > 0),
                                                                        np.logical_and(FeedbackType == 1, contrast_L > 0))]
        RxnTimesTemp = times1stMove_restrTrialsTemp - timesStart_restrTrialsTemp
        timesStart_restrTrials2 = timesStart_restrTrialsTemp[RxnTimesTemp > 0.05]
        times1stMove_restrTrials2 = times1stMove_restrTrialsTemp[RxnTimesTemp > 0.05]
        RxnTimes_restrTrials2 = times1stMove_restrTrials2 - timesStart_restrTrials2  # shoudln't this be Trials2???? MAYO CHANGED

        # The above gives the 2 restriction options, one with correct trials and non-zero contrasts only:
        # timesStart_restrTrials1, times1stMove_restrTrials1, RxnTimes_restrTrials1
        # timesStart_restrTrials2, times1stMove_restrTrials2, RxnTimes_restrTrials2
        # Note: with the above, the np.isnan(RxnTimes) trials should be gone already since RxnTimes>0.05 was applied; double check.

        ### TASK-MODULATION TEST 2 ###
        # The following section is for Left vs. Right pre-movement:
        # Note: Restricted to only correct trials & constrast>0:
        RxnTimes_RightCorrWithNan = RxnTimes[np.logical_and(FeedbackType == 1, contrast_R > 0)]
        RxnTimes_RightCorr = RxnTimes_RightCorrWithNan[~np.isnan(RxnTimes_RightCorrWithNan)]  # not used ATM
        RxnTimes_LeftCorrWithNan = RxnTimes[np.logical_and(FeedbackType == 1, contrast_L > 0)]
        RxnTimes_LeftCorr = RxnTimes_LeftCorrWithNan[~np.isnan(RxnTimes_LeftCorrWithNan)]  # not used ATM
        times1stMove_RightCorr = times1stMove_orig2[np.logical_and(FeedbackType == 1, contrast_R > 0)]
        times1stMove_RightCorr = times1stMove_RightCorr[~np.isnan(RxnTimes_RightCorrWithNan)]
        times1stMove_LeftCorr = times1stMove_orig2[np.logical_and(FeedbackType == 1, contrast_L > 0)]
        times1stMove_LeftCorr = times1stMove_LeftCorr[~np.isnan(RxnTimes_LeftCorrWithNan)]
        # later make more precise by considering the time between L and R

        print(time.time() - start)

        assert np.array_equal(event_times_left, events['eventStim_stL'])  # y
        assert np.array_equal(event_times_right, events['eventStim_stR'])  # y
        assert np.array_equal(event_times_0, events['eventStim_0'])  # y
        assert np.array_equal(event_times_left100, events['eventStim_stL100'])  # y
        assert np.array_equal(event_times_right100, events['eventStim_stR100'])  # y

        assert np.array_equal(event_times_Rchoice, events['eventMove_chR'])
        assert np.array_equal(event_times_Lchoice, events['eventMove_chL'])
        assert np.array_equal(eventMove_stR100chR, events['eventMove_stR100chR'])
        assert np.array_equal(eventMove_stR100chL, events['eventMove_stR100chL'])
        assert np.array_equal(eventMove_stL100chR, events['eventMove_stL100chR'])
        assert np.array_equal(eventMove_stL100chL, events['eventMove_stL100chL'])
        assert np.array_equal(eventMove_stRchR, events['eventMove_stRchR'])
        assert np.array_equal(eventMove_stRchL, events['eventMove_stRchL'])
        assert np.array_equal(eventMove_stLchR, events['eventMove_stLchR'])
        assert np.array_equal(eventMove_stLchL, events['eventMove_stLchL'])
        assert np.array_equal(eventStim_stR100chR, events['eventStim_stR100chR'])
        assert np.array_equal(eventStim_stR100chL, events['eventStim_stR100chL'])
        assert np.array_equal(eventStim_stL100chR, events['eventStim_stL100chR'])
        assert np.array_equal(eventStim_stL100chL, events['eventStim_stL100chL'])

        assert np.array_equal(event_CorrR, events['eventStim_stRfbCorr'])
        assert np.array_equal(event_IncorrR, events['eventStim_stRfbIncorr'])
        assert np.array_equal(event_IncorrL, events['eventStim_stLfbIncorr'])
        assert np.array_equal(event_CorrL, events['eventStim_stLfbCorr'])

        assert np.array_equal(event_FdbckCorr, events['eventFdbck_fbCorr'])
        assert np.array_equal(event_FdbckIncorr, events['eventFdbck_fbIncorr'])

        assert np.array_equal(timesStimOn_CorrTrial, events['eventStim_fbCorr'])
        assert np.array_equal(times1stMove_CorrTrial, events['eventMove_fbCorr'])
        assert np.array_equal(timesStimOn_fullTrial, events['eventStim_All'])
        assert np.array_equal(times1stMove_fullTrial, events['eventMove_All'])

        assert np.array_equal(RxnTimes_CorrTrial, events['rxnTimes_fbCorr'])
        assert np.array_equal(RxnTimes_fullTrial, events['rxnTimes_All'])
        np.testing.assert_equal(RxnTimes, rxn_times)

        assert np.array_equal(timesStart, trials['restr_stimOn'])
        assert np.array_equal(timesStart_restrTrials1, events['eventStimRestr_All'])
        assert np.array_equal(times1stMove_restrTrials1, events['eventMoveRestr_All'])
        assert np.array_equal(timesStart_restrTrials2, events['eventStimRestr_fbCorr'])
        assert np.array_equal(times1stMove_restrTrials2, events['eventMoveRestr_fbCorr'])
        assert np.array_equal(RxnTimes_restrTrials1, events['rxnTimesRestr_All'])
        assert np.array_equal(RxnTimes_restrTrials2, events['rxnTimesRestr_fbCorr'])

        assert np.array_equal(times1stMove_LeftCorr, events['eventMove_stLfbCorr'])
        assert np.array_equal(times1stMove_RightCorr, events['eventMove_stRfbCorr'])
        assert np.array_equal(RxnTimes_RightCorr, events['rxnTimes_stRfbCorr'])
        assert np.array_equal(RxnTimes_LeftCorr, events['rxnTimes_stLfbCorr'])
        # np.testing.assert_equal(RxnTimes_RightCorr, reaction_times['Right_Corr_wNan'])
        # np.testing.assert_equal(RxnTimes_LeftCorr, reaction_times['Left_Corr_wNan'])


        event_times = [event_times_right, event_times_left, event_times_right100, event_times_left100,
                       event_times_0, event_times_Rchoice, event_times_Lchoice,
                       event_CorrR, event_CorrL, event_IncorrR, event_IncorrL,
                       eventMove_stR100chR, eventMove_stR100chL, eventMove_stL100chR, eventMove_stL100chL,
                       eventStim_stR100chR, eventStim_stR100chL, eventStim_stL100chR, eventStim_stL100chL,
                       event_FdbckCorr, event_FdbckIncorr,
                       eventMove_stRchR, eventMove_stRchL, eventMove_stLchL, eventMove_stLchR]


        cluster = clusterIDs[0]
        spike_times_per_cluster = spikes['times'][spikes['clusters'] == cluster]


        data = {}
        # TODO think how to append as list or as array?

        start = time.time()
        firing_rates = {}
        intervals = np.c_[events['eventStim_All'] - 0.2, events['eventStim_All']]
        idx = np.searchsorted(spike_times_per_cluster, intervals)
        data['FR_200msPreStim_All'] = (idx[:, 1] - idx[:, 0]) / 0.2


        intervals = np.c_[events['eventStim_All'],events['eventMove_All']]
        idx = np.searchsorted(spike_times_per_cluster, intervals)
        # account for cases when firstMovement_time before stimOn_time --> set FR to 0 (check this is what we want)
        idx[idx[:, 1] < idx[:, 0], 1] = idx[idx[:, 1] < idx[:, 0], 0]
        data['FR_RxnTime_All'] = (idx[:, 1] - idx[:, 0]) / events['rxnTimes_All']


        intervals = np.c_[events['eventStim_fbCorr'] - 0.2, events['eventStim_fbCorr']]
        idx = np.searchsorted(spike_times_per_cluster, intervals)
        data['FR_200msPreStim_fbCorr'] = (idx[:, 1] - idx[:, 0]) / 0.2


        intervals = np.c_[events['eventStim_fbCorr'], events['eventMove_fbCorr']]
        idx = np.searchsorted(spike_times_per_cluster, intervals)
        # account for cases when firstMovement_time before stimOn_time --> set FR to 0 (check this is what we want)
        idx[idx[:, 1] < idx[:, 0], 1] = idx[idx[:, 1] < idx[:, 0], 0]
        data['FR_RxnTime_fbCorr'] = (idx[:, 1] - idx[:, 0]) / events['rxnTimes_fbCorr']


        intervals = np.c_[events['eventStimRestr_All'] - 0.2, events['eventStimRestr_All']]
        idx = np.searchsorted(spike_times_per_cluster, intervals)
        data['FR_200msPreStim_restr_All'] = (idx[:, 1] - idx[:, 0]) / 0.2


        intervals = np.c_[events['eventStimRestr_All'], events['eventMoveRestr_All']]
        idx = np.searchsorted(spike_times_per_cluster, intervals)
        data['FR_RxnTime_restr_All'] = (idx[:, 1] - idx[:, 0]) / events['rxnTimesRestr_All']


        intervals = np.c_[events['eventStimRestr_fbCorr'] - 0.2, events['eventStimRestr_fbCorr'] ]
        idx = np.searchsorted(spike_times_per_cluster, intervals)
        data['FR_200msPreStim_restr_fbCorr'] = (idx[:, 1] - idx[:, 0]) / 0.2


        intervals = np.c_[events['eventStimRestr_fbCorr'] , events['eventMoveRestr_fbCorr'] ]
        idx = np.searchsorted(spike_times_per_cluster, intervals)
        data['FR_RxnTime_restr_fbCorr'] = (idx[:, 1] - idx[:, 0]) / events['rxnTimesRestr_fbCorr']

        # N>B Key and calculation are not the sameeee!!
        intervals = np.c_[events['eventStim_fbCorr'], events['eventStim_fbCorr'] + 0.4]
        idx = np.searchsorted(spike_times_per_cluster, intervals)
        data['FR_400msPostStim_restr_fbCorr'] = (idx[:, 1] - idx[:, 0]) / 0.4

        # N>B Key and calculation are not the sameeee!!
        intervals = np.c_[events['eventStim_fbCorr'] + 0.05, events['eventStim_fbCorr'] + 0.15]
        idx = np.searchsorted(spike_times_per_cluster, intervals)
        data['FR_100msPostStim_restr_fbCorr'] = (idx[:, 1] - idx[:, 0]) / 0.1

        # N>B Key and calculation are not the sameeee!!
        intervals = np.c_[events['eventMove_stRfbCorr'] - 0.1, events['eventMove_stRfbCorr'] + 0.05]
        idx = np.searchsorted(spike_times_per_cluster, intervals)
        data['FR_RightPreMove_stRfbCorr'] = (idx[:, 1] - idx[:, 0]) / 0.15

        intervals = np.c_[events['eventMove_stLfbCorr'] - 0.1, events['eventMove_stLfbCorr'] + 0.05]
        idx = np.searchsorted(spike_times_per_cluster, intervals)
        data['FR_LeftPreMove_stLfbCorr'] = (idx[:, 1] - idx[:, 0]) / 0.15

        print(time.time() - start)

        start = time.time()
        FR_200msBeforeStim_perClust = [sum(np.bitwise_and(spike_times_per_cluster >= timesStimOn_fullTrial[t]-0.2, spike_times_per_cluster < timesStimOn_fullTrial[t]))/0.2
                             for t in range(0,len(timesStimOn_fullTrial))]
        FR_during_RxnTime_perClust = [sum(np.bitwise_and(spike_times_per_cluster >= timesStimOn_fullTrial[t], spike_times_per_cluster <= times1stMove_fullTrial[t]))/RxnTimes_fullTrial[t]
                             for t in range(0,len(timesStimOn_fullTrial))]
        FR_200msPreStim_Corr = [sum(np.bitwise_and(spike_times_per_cluster >= timesStimOn_CorrTrial[t]-0.2, spike_times_per_cluster < timesStimOn_CorrTrial[t]))/0.2
                             for t in range(0,len(timesStimOn_CorrTrial))]
        FR_during_RxnTime_Corr = [sum(np.bitwise_and(spike_times_per_cluster >= timesStimOn_CorrTrial[t], spike_times_per_cluster <= times1stMove_CorrTrial[t]))/RxnTimes_CorrTrial[t]
                             for t in range(0,len(timesStimOn_CorrTrial))]
        FR_200msPreStim_restrTrials1 = [sum(np.bitwise_and(spike_times_per_cluster >= timesStart_restrTrials1[t]-0.2, spike_times_per_cluster < timesStart_restrTrials1[t]))/0.2
                             for t in range(0,len(timesStart_restrTrials1))]
        FR_RxnTime_perClust_restrTrials1 = [sum(np.bitwise_and(spike_times_per_cluster >= timesStart_restrTrials1[t], spike_times_per_cluster <= times1stMove_restrTrials1[t]))/RxnTimes_restrTrials1[t]
                             for t in range(0,len(timesStart_restrTrials1))]
        FR_200msPreStim_restrTrials2 = [sum(np.bitwise_and(spike_times_per_cluster >= timesStart_restrTrials2[t]-0.2, spike_times_per_cluster < timesStart_restrTrials2[t]))/0.2
                             for t in range(0,len(timesStart_restrTrials2))]
        FR_RxnTime_perClust_restrTrials2 = [sum(np.bitwise_and(spike_times_per_cluster >= timesStart_restrTrials2[t], spike_times_per_cluster <= times1stMove_restrTrials2[t]))/RxnTimes_restrTrials2[t]
                             for t in range(0,len(timesStart_restrTrials2))]
        # N>B Key and calculation are not the sameeee!!
        FR_400msPostStim_restrTrials1 = [sum(np.bitwise_and(spike_times_per_cluster >= timesStimOn_CorrTrial[t], spike_times_per_cluster < (timesStimOn_CorrTrial[t] + 0.4)))/0.4
                             for t in range(0,len(timesStimOn_CorrTrial))]
        # N>B Key and calculation are not the sameeee!!
        FR_100msPostStim_restrTrials1 = [sum(np.bitwise_and(spike_times_per_cluster >= timesStimOn_CorrTrial[t]+0.05, spike_times_per_cluster < (timesStimOn_CorrTrial[t] + 0.15)))/0.1
                             for t in range(0,len(timesStimOn_CorrTrial))]
        FR_RightPreMove_restrTrials1 = [sum(np.bitwise_and(spike_times_per_cluster >= times1stMove_RightCorr[t]-0.1, spike_times_per_cluster < (times1stMove_RightCorr[t] + 0.05)))/0.15
                             for t in range(0,len(times1stMove_RightCorr))]
        FR_LeftPreMove_restrTrials1 = [sum(np.bitwise_and(spike_times_per_cluster >= times1stMove_LeftCorr[t]-0.1, spike_times_per_cluster < (times1stMove_LeftCorr[t] + 0.05)))/0.15
                             for t in range(0,len(times1stMove_LeftCorr))]

        print(time.time() - start)

        assert np.array_equal(data['FR_200msPreStim_All'], FR_200msBeforeStim_perClust)
        assert np.array_equal(data['FR_RxnTime_All'], FR_during_RxnTime_perClust)
        assert np.array_equal(data['FR_200msPreStim_fbCorr'], FR_200msPreStim_Corr)
        assert np.array_equal(data['FR_RxnTime_fbCorr'], FR_during_RxnTime_Corr)
        assert np.array_equal(data['FR_200msPreStim_restr_All'], FR_200msPreStim_restrTrials1)
        assert np.array_equal(data['FR_RxnTime_restr_All'], FR_RxnTime_perClust_restrTrials1)
        assert np.array_equal(data['FR_200msPreStim_restr_fbCorr'], FR_200msPreStim_restrTrials2)
        assert np.array_equal(data['FR_RxnTime_restr_fbCorr'], FR_RxnTime_perClust_restrTrials2)
        assert np.array_equal(data['FR_400msPostStim_restr_fbCorr'], FR_400msPostStim_restrTrials1)
        assert np.array_equal(data['FR_100msPostStim_restr_fbCorr'], FR_100msPostStim_restrTrials1)
        assert np.array_equal(data['FR_RightPreMove_stRfbCorr'], FR_RightPreMove_restrTrials1)
        assert np.array_equal(data['FR_LeftPreMove_stLfbCorr'], FR_LeftPreMove_restrTrials1)


        # MARSA CODE
        start = time.time()
        event_times = [event_times_right, event_times_left, event_times_right100, event_times_left100,
                       event_times_0, event_times_Rchoice, event_times_Lchoice,
                       event_CorrR, event_CorrL, event_IncorrR, event_IncorrL,
                       eventMove_stR100chR, eventMove_stR100chL, eventMove_stL100chR, eventMove_stL100chL,
                       eventStim_stR100chR, eventStim_stR100chL, eventStim_stL100chR, eventStim_stL100chL,
                       event_FdbckCorr, event_FdbckIncorr,
                       eventMove_stRchR, eventMove_stRchL, eventMove_stLchL, eventMove_stLchR]

        TimeVect_FR = []
        TimeVect_FF = []
        maskNoEvent = [1] * len(event_times)  # []
        for idx in range(0, len(event_times)):
            if size(event_times[idx]) == 0:
                maskNoEvent[idx] = event_times[idx]  # 0 #1 #.append(1)
                event_times[idx] = np.array([1], dtype=float64)

        # Calculate the FR using a smaller bin size of binSzFRPeri (~20 ms) and with sliding window for 1 cluster at a time, for all events:
        ActivitySmallBin = [cluster_peths_FR_FF_sliding(spikes['times'][spikes['clusters'] == cluster],
                                                        event_times[x], pre_time=pre_time, post_time=post_time,
                                                        hist_win=binSzFRPeri, N_SlidesPerWind=n_slideFR, causal=Caus) for x in
                            range(0, len(event_times))]
        FRoverT = [ActivitySmallBin[x][0] for x in range(0, len(event_times))]
        FR_STD_overT = [ActivitySmallBin[x][1] for x in range(0, len(event_times))]
        TimeVect_FR0 = ActivitySmallBin[0][3]  # the 0th event, the 4th np.array (with idx=3) which is the time vector
        TimeVect_FR.append(TimeVect_FR0)

        # Calculate the FF using a larger bin size of binSzFFPeri (~100 ms) and with sliding window for 1 cluster at a time, for all events:
        ActivitySlideLargeBin = [cluster_peths_FR_FF_sliding(spikes['times'][spikes['clusters'] == cluster],
                                                             event_times[x], pre_time=pre_time, post_time=post_time,
                                                             hist_win=binSzFFPeri, N_SlidesPerWind=n_slideFF, causal=Caus) for x
                                 in range(0, len(event_times))]
        FFoverT = [ActivitySlideLargeBin[x][2] for x in range(0, len(event_times))]
        TimeVect_FF0 = ActivitySlideLargeBin[0][3]  # the 0th event, the 4th np.array (with idx=3) which is the time vector
        TimeVect_FF.append(TimeVect_FF0)

        # test=[d for d, s in zip(FRoverT, maskNoEvent) if s] #keep the FRoverT cases where maskNoEvent was true, i.e., 1
        for idx in range(0, len(event_times)):
            if size(maskNoEvent[idx]) == 0:  # cases where there were no events
                event_times[idx] = maskNoEvent[idx]
                # FFoverT[idx] = np.empty([1, 1, int((post_time - (-pre_time))/binSzFFPeri)])
                FFoverT[idx][:] = np.NaN
                # FRoverT[idx] = np.empty([1, int((post_time - (-pre_time))/binSzFRPeri)])
                FRoverT[idx][:] = np.NaN
                FR_STD_overT[idx] = FRoverT[idx]

        FR_PreEvent = [float(np.nanmean(FRoverT[x][TimeVect_FR0<0])) for x in range(0,len(event_times))]
        FF_PreEvent = [float(np.nanmean(FFoverT[x][TimeVect_FF0<0])) for x in range(0,len(event_times))]
        FR_PostEvent = [float(np.nanmean(FRoverT[x][np.logical_and(TimeVect_FR0>0, TimeVect_FR0<CapPostTime)])) for x in range(0,len(event_times))]
        FF_PostEvent = [float(np.nanmean(FFoverT[x][np.logical_and(TimeVect_FF0>0, TimeVect_FF0<CapPostTime)])) for x in range(0,len(event_times))]

        print(time.time() - start)

        # MAYO CODE
        start = time.time()
        event_keys = ['eventStim_stR', 'eventStim_stL', 'eventStim_stR100', 'eventStim_stL100', 'eventStim_0',
                      'eventMove_chR', 'eventMove_chL', 'eventStim_stRfbCorr', 'eventStim_stLfbCorr', 'eventStim_stRfbIncorr',
                      'eventStim_stLfbIncorr', 'eventStim_stR100chR', 'eventStim_stR100chL', 'eventStim_stL100chR',
                      'eventStim_stL100chL', 'eventMove_stR100chR', 'eventMove_stR100chL', 'eventMove_stL100chR',
                      'eventMove_stL100chL', 'eventMove_stRchR', 'eventMove_stRchL', 'eventMove_stLchL', 'eventMove_stLchR',
                      'eventFdbck_fbCorr', 'eventFdbck_fbIncorr']


        data = {}
        # TODO get this working so we don't rely on
        # FR_len = (pre_time + post_time + binSzFRPeri/n_slideFR) / (binSzFRPeri/n_slideFR)
        # FF_len = (pre_time + post_time + binSzFFPeri/n_slideFF) / (binSzFFPeri/n_slideFF)
        for iK, key in enumerate(event_keys):
            data_temp = {}
            if events[key].shape[0] == 0:
                # need to think about this
                data_temp['FR'] = np.nan * np.ones_like(time_FR)
                data_temp['FR_std'] = np.nan * np.ones_like(time_FR)
                data_temp['FF'] = np.nan * np.ones_like(time_FF)
                data_temp['FR_PreTime'] = np.nan
                data_temp['FR_PostTime'] = np.nan
                data_temp['FF_PreTime'] = np.nan
                data_temp['FF_PostTime'] = np.nan

            else:
                # Do we really need to save these?
                data_temp['FR'], data_temp['FR_std'], _, time_FR = \
                    cluster_peths_FR_FF_sliding(spike_times_per_cluster, events[key], pre_time=pre_time, post_time=post_time,
                                                hist_win=binSzFRPeri, N_SlidesPerWind=n_slideFR, causal=Caus)
                _, _, data_temp['FF'], time_FF = \
                    cluster_peths_FR_FF_sliding(spike_times_per_cluster, events[key], pre_time=pre_time, post_time=post_time,
                                                hist_win=binSzFFPeri, N_SlidesPerWind=n_slideFF, causal=Caus)

                data_temp['FR_PreEvent'] = np.nanmean(data_temp['FR'][time_FR < 0])
                data_temp['FF_PreEvent'] = np.nanmean(data_temp['FF'][time_FF < 0])
                data_temp['FR_PostEvent'] = np.nanmean(data_temp['FR'][np.logical_and(time_FR >0, time_FR < CapPostTime)])
                data_temp['FF_PostEvent'] = np.nanmean(data_temp['FF'][np.logical_and(time_FF >0, time_FF < CapPostTime)])

                # case where ik = 0 has no trials ( do we need to take into account? NO)
                if iK == 0:
                    # see whether we want to save this for each event or not?
                    data['Time_FR'] = time_FR
                    data['Time_FF'] = time_FF

        print(time.time() - start)
        for ik, key in enumerate(event_keys):
            assert np.array_equal(data[key]['FR'], FRoverT[ik])
            assert np.array_equal(data[key]['FR_std'], FR_STD_overT)
            assert np.array_equal(data[key]['FF'], FFoverT)
            assert np.array_equal(data[key]['FR_PreEvent'], FR_PreEvent)
            assert np.array_equal(data[key]['FR_PostEvent'], FR_PreEvent)
            assert np.array_equal(data[key]['FF_PreEvent'], FF_PreEvent)
            assert np.array_equal(data[key]['FF_PostEvent'], FF_PostEvent)
            assert np.array_equal(data['Time_FR'], TimeVect_FR0)
            assert np.array_equal(data['Time_FF'], TimeVect_FF0)






    except AssertionError as err:
        print('ERRRORED')
        print(traceback.format_exc())
        print(err)
        print(pid)





def cluster_peths_FR_FF_sliding(ts, align_times, pre_time=0.2, post_time=0.5,
                                hist_win=0.1, N_SlidesPerWind=5, causal=0):
    """
    Calcluate peri-event time histograms of one unit/cluster at a time; returns
    means and standard deviations of FR and FF over time for each time point
    using a sliding window.

    :param ts: spike times of cluster (in seconds)
    :type ts: array-like
    :param align_times: times (in seconds) to align peths to
    :type align_times: array-like
    :param pre_time: time (in seconds) to precede align times in peth
    :type pre_time: float
    :param post_time: time (in seconds) to follow align times in peth
    :type post_time: float
    :param hist_win: width of time windows (in seconds) to bin spikes for each sliding window
    :type hist_win: float
    :param N_SlidesPerWind: The # of slides to do within each histogram window, i.e. increase in time resolution
    :type N_SlidesPerWind: float
    :param causal: whether or not to place time points at the end of each hist_win (1) or in the center (0)
    :type causal: float
    :return: FR_sorted, FR_STD_sorted, FF_sorted, TimeVect_sorted
    :rtype: np.array
    """
    # slidingStep = bin_size/n_slides
    epoch = np.r_[-1 * pre_time, post_time]
    tshift = hist_win / N_SlidesPerWind
    if causal == 1:  # Place time points at the end of each hist_win, i.e., only past events are taken into account.
        epoch[0] = epoch[0] - hist_win / 2  # to start earlier since we're shifting the time later
        # epoch[1] = epoch[1] - hist_win/2

    FR_unsort, FRstd_unsort, FF_unsort, TimeVect = [], [], [], []
    for s in range(N_SlidesPerWind):
        ts_shift = ts[ts > (s * tshift)]
        PethsPerShift, BinnedSpPerShift = calculate_peths(ts_shift, np.ones_like(ts_shift), np.array([1]),
                                                          (align_times + s * tshift), pre_time=abs(epoch[0]),
                                                          post_time=(epoch[1] - s * tshift), bin_size=hist_win,
                                                          smoothing=0, return_fr=False)
        # To Do: The post_time=(epoch[1]- s*tshift) might need to become post_time=(epoch[1]- s*hist_win) or similar.

        CountPerBinPerShift = BinnedSpPerShift.reshape(BinnedSpPerShift.shape[0], BinnedSpPerShift.shape[2])
        # FR_PerTrialPerShift = CountPerBinPerShift/hist_win
        FR_TrialAvgPerShift = np.nanmean(CountPerBinPerShift, axis=0) / hist_win
        FR_TrialSTDPerShift = np.nanstd(CountPerBinPerShift,
                                        axis=0) / hist_win  # stdev of firing rate; same as np.std(CountPerBinPerShift/hist_win, axis=0)
        FF_PerShift = np.nanvar(CountPerBinPerShift, axis=0) / np.nanmean(CountPerBinPerShift, axis=0)
        TimeVect_PerShift = PethsPerShift['tscale'] + s * tshift  # np.arange(FR_PerShift.size) * hist_win + tshift*s #per slide

        # Place time points at the end of each hist_win (causal = 1),i.e., only past events are taken into account.
        # Otherwise, time points are at the center of the time bins (using 'calculate_peths')
        if causal == 1:
            TimeVect_PerShift = TimeVect_PerShift + hist_win / 2

        # Append this shifted result with previous ones:
        FR_unsort = np.hstack((FR_unsort, FR_TrialAvgPerShift))
        FRstd_unsort = np.hstack((FRstd_unsort, FR_TrialSTDPerShift))
        FF_unsort = np.hstack((FF_unsort, FF_PerShift))
        TimeVect = np.hstack((TimeVect, TimeVect_PerShift))  # stacks the time vectors

    # Sort the time and FR vectors and convert lists to an np.array:
    TimeVect_sorted = np.sort(TimeVect)
    FR_sorted = np.array([x for _, x in sorted(zip(TimeVect, FR_unsort))])
    FR_STD_sorted = np.array([x for _, x in sorted(zip(TimeVect, FRstd_unsort))])
    FF_sorted = np.array([x for _, x in sorted(zip(TimeVect, FF_unsort))])

    return FR_sorted, FR_STD_sorted, FF_sorted, TimeVect_sorted

import numpy as np


def get_event_times(trials):
    # Make a dict of all trials permutations that we will use later for analysis
    # fb - feedback, either Corr (correct) or Incorr (incorrect)
    # st - stimulus side, either L (left) or R (right)
    # ch - choice, either L (leftward wheel motion (1)) or R (rightward wheel motion (-1))
    # number - stimulus contrast percentage


    nanStim = np.isnan(trials['stimOn_times'])
    nanMovement = np.isnan(trials['firstMovement_times'])
    nanStimMovement = np.bitwise_or(nanStim, nanMovement)
    nanFeedback = np.isnan(trials['feedback_times'])

    print(f'stimOn_nans: {np.sum(nanStim)}')
    print(f'movement_nans: {np.sum(nanMovement)}')
    print(f'feedback_nans: {np.sum(nanFeedback)}')
    print(f'stim and movement_nans: {np.sum(nanStimMovement)}')

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
    events['eventStim_stRfbIncorr'] = trials['stimOn_times'][np.bitwise_and(trials_outcome['stR_fbIncorr'], ~nanStim)]
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
    events['eventMove_stLfbIncorr'] = trials['firstMovement_times'][
        np.bitwise_and(trials_outcome['stL_fbIncorr'], ~nanStimMovement)]
    events['eventMove_stRfbCorr'] = trials['firstMovement_times'][np.bitwise_and(trials_outcome['stR_fbCorr'], ~nanStimMovement)]
    events['eventMove_stRfbIncorr'] = trials['firstMovement_times'][
        np.bitwise_and(trials_outcome['stR_fbIncorr'], ~nanStimMovement)]
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

    # TIMEWARPED TIMES
    trials['TW_stimOn'] = np.copy(trials['stimOn_times'])
    trials['TW_stimOn'][rxn_times > 0.2] = trials['firstMovement_times'][rxn_times > 0.2] - 0.2
    events['eventStimTW_All'] = trials['TW_stimOn'][np.bitwise_and(rxn_times > 0.05, ~nanStimMovement)]
    events['eventStimTW_fbCorr'] = trials['TW_stimOn'][np.bitwise_and(np.bitwise_and(trials_outcome['fbCorr_gt0'],
                                                                                     rxn_times > 0.05), ~nanStimMovement)]
    events['eventMoveTW_All'] = trials['firstMovement_times'][np.bitwise_and(rxn_times > 0.05, ~nanStimMovement)]
    events['eventMoveTW_fbCorr'] = trials['firstMovement_times'][np.bitwise_and(np.bitwise_and(trials_outcome['fbCorr_gt0'],
                                                                                rxn_times > 0.05), ~nanStimMovement)]
    events['rxnTimesTW_All'] = events['eventMoveTW_All'] - events['eventStimTW_All']
    events['rxnTimesTW_fbCorr'] = events['eventMoveTW_fbCorr'] - events['eventStimTW_fbCorr']

    return events
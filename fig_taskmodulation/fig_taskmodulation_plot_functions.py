"""
@author: Marsa Taheri, extended from Sebastian's code
"""

import numpy as np
import matplotlib.pyplot as plt

from one.api import ONE
from iblatlas.atlas import AllenAtlas
from brainbox.io.one import SpikeSortingLoader
from brainbox.task.trials import find_trial_ids
from reproducible_ephys_processing import compute_psth,  compute_psth_rxn_time
import seaborn as sns

# Defaults parameters for psth computation
default_params = {'fr_bin_size': 0.01,
                  'ff_bin_size': 0.1,
                  'align_event': 'move',
                  'event_epoch': [-0.4, 0.22],
                  'base_event': None,
                  'base_epoch': None,
                  'norm': None,
                  'smoothing': None,
                  'slide_kwargs_ff': {'n_win': 3, 'causal': 1},
                  'slide_kwargs_fr': {'n_win': 2, 'causal': 1},
                  'kernel_kwargs': {'kernel': None}}


def plot_raster_and_psth_LvsR(pid, neuron, contrasts=(1, 0.25, 0.125, 0.0625, 0), feedback='all',
                              ax=None, one=None, ba=None, plot_ff=False, **kwargs):

    # Distingishes neural activity between L and R choices (Not different stim. contrast levels)
    one = one or ONE()
    ba = ba or AllenAtlas()

    fr_bin_size = kwargs.get('fr_bin_size', default_params['fr_bin_size'])
    ff_bin_size = kwargs.get('ff_bin_size', default_params['ff_bin_size'])
    align_event = kwargs.get('align_event', default_params['align_event'])
    event_epoch = kwargs.get('event_epoch', default_params['event_epoch'])
    base_event = kwargs.get('base_event', default_params['base_event'])
    base_epoch = kwargs.get('base_epoch', default_params['base_epoch'])
    norm = kwargs.get('norm', default_params['norm'])
    smoothing = kwargs.get('smoothing', default_params['smoothing'])
    slide_kwargs_fr = kwargs.get('slide_kwargs_fr', default_params['slide_kwargs_fr'])
    slide_kwargs_ff = kwargs.get('slide_kwargs_ff', default_params['slide_kwargs_ff'])
    kernel_kwargs = kwargs.get('kernel_kwargs', default_params['kernel_kwargs'])

    figsize = kwargs.get('figsize', (9, 12))
    zero_line_c = kwargs.get('zero_line_c', 'k')

    eid, probe = one.pid2eid(pid)
    sl = SpikeSortingLoader(eid=eid, pname=probe, one=one, atlas=ba)
    spikes, clusters, channels = sl.load_spike_sorting(revision='2024-03-22', enforce_version=False)

    spike_idx = np.isin(spikes['clusters'], neuron)
    if np.sum(spike_idx) == 0:
        print(f'Warning no spikes detected for insertion: {pid} and neuron: {neuron}')

    trials = one.load_object(eid, 'trials', collection='alf')
    # Remove trials with nans in the stimOn_times or the firstMovement_times
    nanStimMove = np.bitwise_or(np.isnan(trials['stimOn_times']), np.isnan(trials['firstMovement_times']))
    trials = {key: trials[key][~nanStimMove] for key in trials.keys()}

    # left choice -> when stim left, feedback true + stim_right, feedback false
    trial_id, _ = find_trial_ids(trials, side='all', choice=feedback, contrast=contrasts)
    trial_mask = np.zeros((trials['stimOn_times'].size)).astype(bool)
    trial_mask[trial_id] = True

    trials = {key: trials[key][trial_mask] for key in trials.keys()}
    trials['contrast'] = np.nansum([trials['contrastLeft'], trials['contrastRight']], axis=0)

    if align_event == 'move':  # To align to movement times:
        eventTimes = trials['firstMovement_times']
    elif align_event == 'stim':
        eventTimes = trials['stimOn_times']

    if base_event == 'move':
        eventBase = trials['firstMovement_times']
    elif base_event == 'stim':
        eventBase = trials['stimOn_times']
    else:
        eventBase = None

    if ax is None:
        if plot_ff:
            fig, ax = plt.subplots(3, 1, figsize=figsize)
        else:
            fig, ax = plt.subplots(2, 1, figsize=figsize)

    boundary_width = 0.01
    base_grey = 0.3
    counter = 0
    count_list = [0]
    ylabel_pos = []
    pre_time = event_epoch[0]
    post_time = event_epoch[1]

    # Plot the individual spikes
    for ch in [-1, 1]:
        # When only correct feedback, then ch=1 means left side stim and choice
        events = eventTimes[trials['choice'] == ch]
        for i, time in enumerate(events):
            idx = np.bitwise_and(spikes.times[spike_idx] >= time + pre_time, spikes.times[spike_idx] <= time + post_time)
            ax[0].vlines(spikes.times[spike_idx][idx] - time, counter + i, counter + i + 1, color='k')
        counter += len(events)
        count_list.append(counter)
        ax[0].set_xlim(pre_time - fr_bin_size / 2, post_time + fr_bin_size / 2)

    # Plot the bar indicating stim/choice side on the left side of figure
    for i, ch in enumerate([-1, 1]):
        # Determine color of the colorbar
        if ch == -1:
            ch_color = 1
        elif ch == 1:
            ch_color = 0.1  # Since no alpha here, we adjust this number from 0.45
        top = count_list[i]
        bottom = count_list[i + 1]
        # Position of the contrast colorbar:
        ax[0].fill_between([pre_time - fr_bin_size / 2, pre_time - fr_bin_size / 2 + boundary_width],
                            [top, top], [bottom, bottom], color=str(1 - (base_grey + ch_color * (1 - base_grey))))
        ylabel_pos.append((top - bottom) / 2 + bottom)

    ax[0].set_yticks(ylabel_pos)
    ax[0].set_yticklabels(['Right', 'Left'])
    ax[0].axvline(0, color=zero_line_c, ls='--')
    ax[0].set_ylim(0, counter)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['left'].set_visible(False)
    ax[0].spines['bottom'].set_visible(False)
    ax[0].tick_params(left=False, right=False, labelbottom=False, bottom=False)

    # Compute the psths for firing rate for each contrast
    for ch in [-1, 1]:
        # Determine color of the trace
        if ch == -1:
            ch_color = 1
        elif ch == 1:
            ch_color = 0.45
        # When only correct feedback, then ch=1 means left side stim and choice
        events = eventTimes[trials['choice'] == ch]
        fr, fr_std, t = compute_psth(spikes['times'][spike_idx], spikes['clusters'][spike_idx], np.array([neuron]),
                                     events, align_epoch=event_epoch, bin_size=fr_bin_size,
                                     baseline_events=eventBase, base_epoch=base_epoch, smoothing=smoothing, norm=norm,
                                     slide_kwargs=slide_kwargs_fr, kernel_kwargs=kernel_kwargs)
        ax[1].plot(t, fr[0], c=str(1 - ch_color*(base_grey + (1 - base_grey))))
        ax[1].fill_between(t, fr[0] + fr_std[0] / np.sqrt(len(events)), fr[0] - fr_std[0] / np.sqrt(len(events)),
                           color=str(1 - (base_grey + ch_color*(1 - base_grey))), alpha=0.3)
        ax[1].set_xlim(left=pre_time, right=post_time)

    ax[1].axvline(0, color=zero_line_c, ls='--')
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].set_ylabel("Firing rate (sp/s)")
    ax[1].set_xlim(left=pre_time, right=post_time)
    ax[1].set_xticks([-0.15, 0, 0.15])

    # Optionally plot the fanofactor
    if plot_ff:
        for ch in [-1, 1]:
            # Determine color of the trace
            if ch == 1:
                ch_color = 0.35
            elif ch == -1:
                ch_color = 1
            # When only correct feedback, then ch=1 means left side stim and choice
            events = eventTimes[trials['choice'] == ch]
            _, _, ff, t = compute_psth(spikes['times'][spike_idx], spikes['clusters'][spike_idx], np.array([neuron]),
                                       events, align_epoch=event_epoch, bin_size=ff_bin_size,
                                       baseline_events=eventBase, base_epoch=base_epoch, smoothing=smoothing, norm=norm,
                                       slide_kwargs=slide_kwargs_ff, return_ff=True)
            ax[2].plot(t, ff[0],ch_color=str(1 - (base_grey + ch_color * (1 - base_grey))))

        ax[2].axvline(0, color=zero_line_c, ls='--')
        ax[2].spines['right'].set_visible(False)
        ax[2].spines['top'].set_visible(False)
        ax[2].set_ylabel("Fano Factor")
        ax[2].set_xlim(left=pre_time, right=post_time)
        if align_event == 'move':
            ax[2].set_xlabel("Time from movement onset (s)")
        else:
            ax[2].set_xlabel("Time from stimulus onset (s)")
    else:
        if align_event == 'move':
            ax[1].set_xlabel("Time from movement onset (s)")
        else:
            ax[1].set_xlabel("Time from stimulus onset (s)")

    return ax


def plot_raster_and_psth(pid, neuron, contrasts=(1, 0.25, 0.125, 0.0625, 0), side='all', feedback='all',
                         ax=None, one=None, ba=None, plot_ff=False, rxn_time=False, **kwargs):
    #distingishes neural activity between different stim. contrast levels
    one = one or ONE()
    ba = ba or AllenAtlas()

    fr_bin_size = kwargs.get('fr_bin_size', default_params['fr_bin_size'])
    ff_bin_size = kwargs.get('ff_bin_size', default_params['ff_bin_size'])
    align_event = kwargs.get('align_event', default_params['align_event'])
    event_epoch = kwargs.get('event_epoch', default_params['event_epoch'])
    base_event = kwargs.get('base_event', default_params['base_event'])
    base_epoch = kwargs.get('base_epoch', default_params['base_epoch'])
    norm = kwargs.get('norm', default_params['norm'])
    smoothing = kwargs.get('smoothing', default_params['smoothing'])
    slide_kwargs_fr = kwargs.get('slide_kwargs_fr', default_params['slide_kwargs_fr'])
    slide_kwargs_ff = kwargs.get('slide_kwargs_ff', default_params['slide_kwargs_ff'])
    kernel_kwargs = kwargs.get('kernel_kwargs', default_params['kernel_kwargs'])

    figsize = kwargs.get('figsize', (9, 12))
    labelsize = kwargs.get('labelsize', 8)
    zero_line_c = kwargs.get('zero_line_c', 'k')

    eid, probe = one.pid2eid(pid)
    sl = SpikeSortingLoader(eid=eid, pname=probe, one=one, atlas=ba)
    spikes, clusters, channels = sl.load_spike_sorting(revision='2024-03-22', enforce_version=False)

    spike_idx = np.isin(spikes['clusters'], neuron)
    if np.sum(spike_idx) == 0:
        print('warning, warning')

    trials = one.load_object(eid, 'trials', collection='alf')
    # Remove trials with nans in the stimOn_times or the firstMovement_times
    nanStimMove = np.bitwise_or(np.isnan(trials['stimOn_times']), np.isnan(trials['firstMovement_times']))
    trials = {key: trials[key][~nanStimMove] for key in trials.keys()}

    # left choice -> when stim left, feedback true + stim_right, feedback false
    trial_id, _ = find_trial_ids(trials, side=side, choice=feedback, contrast=contrasts)
    trial_mask = np.zeros((trials['stimOn_times'].size)).astype(bool)
    trial_mask[trial_id] = True

    trials = {key: trials[key][trial_mask] for key in trials.keys()}
    trials['contrast'] = np.nansum([trials['contrastLeft'], trials['contrastRight']], axis=0)

    if align_event == 'move':  # To align to movement times:
        eventTimes = trials['firstMovement_times']
    elif align_event == 'stim':
        eventTimes = trials['stimOn_times']

    if base_event == 'move':
        eventBase = trials['firstMovement_times']
    elif base_event == 'stim':
        eventBase = trials['stimOn_times']
    else:
        eventBase = None

    if ax is None:
        if plot_ff:
            fig, ax = plt.subplots(3, 1, figsize=figsize)
        else:
            fig, ax = plt.subplots(2, 1, figsize=figsize)

    boundary_width = 0.01
    base_grey = 0.3
    counter = 0
    contrast_count_list = [0]
    ylabel_pos = []
    contrasts = np.sort(contrasts)
    pre_time = event_epoch[0]
    post_time = event_epoch[1]

    # Plot the individual spikes
    for c in contrasts:
        events = eventTimes[trials['contrast'] == c]

        if rxn_time==True:
            #Find the rxn time intervals:
            eventTimes_Stim = trials['stimOn_times']
            #eventTimes_Move = trials['firstMovement_times']
            eventsStim = eventTimes_Stim[trials['contrast'] == c]
            #eventsMove = eventTimes_Move[trials['contrast'] == c]

            if align_event == 'move':
                post_time_adjusted = 0.2
                #If needed, also remove the trials that will not be included, i.e., rxn time<50 ms (MT)
                for i, time in enumerate(events):
                    pre_time_adjusted = - min(time - eventsStim[i], 0.2)

                    idx = np.bitwise_and(spikes.times[spike_idx] >= time + pre_time_adjusted, spikes.times[spike_idx] <= time + post_time_adjusted)
                    ax[0].vlines(spikes.times[spike_idx][idx] - time, counter + i, counter + i + 1, color='k')
                counter += len(events)
                contrast_count_list.append(counter)

                ax[0].set_xlim(-0.2 - fr_bin_size / 2, post_time_adjusted + fr_bin_size / 2)

            else:
                post_time_adjusted = 0.1
                pre_time_adjusted = -0.2
                for i, time in enumerate(events):
                    idx = np.bitwise_and(spikes.times[spike_idx] >= time + pre_time_adjusted, spikes.times[spike_idx] <= time + post_time_adjusted)
                    ax[0].vlines(spikes.times[spike_idx][idx] - time, counter + i, counter + i + 1, color='k')
                counter += len(events)
                contrast_count_list.append(counter)

                ax[0].set_xlim(pre_time_adjusted - fr_bin_size / 2, post_time_adjusted + fr_bin_size / 2)

        else:
            for i, time in enumerate(events):
                idx = np.bitwise_and(spikes.times[spike_idx] >= time + pre_time, spikes.times[spike_idx] <= time + post_time)
                ax[0].vlines(spikes.times[spike_idx][idx] - time, counter + i, counter + i + 1, color='k')
            counter += len(events)
            contrast_count_list.append(counter)

            ax[0].set_xlim(pre_time - fr_bin_size / 2, post_time + fr_bin_size / 2)


    # Plot the contrast bar at the side of figure
    for i, c in enumerate(contrasts):
        top = contrast_count_list[i]
        bottom = contrast_count_list[i + 1]
        # Position of the contrast colorbar:
        if rxn_time==True: #and align_event == 'move':
            ax[0].fill_between([-0.2 - fr_bin_size / 2, -0.2 - fr_bin_size / 2 + boundary_width],
                               [top, top], [bottom, bottom], zorder=3, color=str(1 - (base_grey + c * (1 - base_grey))))
        else:
            ax[0].fill_between([pre_time - fr_bin_size / 2, pre_time - fr_bin_size / 2 + boundary_width],
                               [top, top], [bottom, bottom], zorder=3, color=str(1 - (base_grey + c * (1 - base_grey))))
        ylabel_pos.append((top - bottom) / 2 + bottom)

    ax[0].set_yticks(ylabel_pos)
    ax[0].set_yticklabels(contrasts)
    ax[0].axvline(0, color=zero_line_c, ls='--')
    #ax[0].set_xlim(pre_time - fr_bin_size / 2, post_time + fr_bin_size / 2)
    ax[0].set_ylim(0, counter)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['left'].set_visible(False)
    ax[0].spines['bottom'].set_visible(False)
    ax[0].tick_params(left=False, right=False, labelbottom=False, bottom=False)  # , labelsize=labelsize)
    # ax[0].set_title("Contrast", loc='left')  # , size=labelsize)

    # Comppute the psths for firing rate for each contrast
    for c in contrasts:
        events = eventTimes[trials['contrast'] == c]
        if rxn_time==True:
            #Find the rxn time intervals:
            eventTimes_Stim = trials['stimOn_times']
            eventTimes_Move = trials['firstMovement_times']
            eventsStim = eventTimes_Stim[trials['contrast'] == c]
            eventsMove = eventTimes_Move[trials['contrast'] == c]


            # fr, fr_std, t = compute_psth(spikes['times'][spike_idx], spikes['clusters'][spike_idx], np.array([neuron]),
            #                              events, align_epoch=event_epoch, bin_size=fr_bin_size,
            #                              baseline_events=eventBase, base_epoch=base_epoch, smoothing=smoothing, norm=norm,
            #                              slide_kwargs=slide_kwargs_fr, kernel_kwargs=kernel_kwargs)

            if align_event == 'move':
                fr, fr_std, t = compute_psth_rxn_time(spikes['times'][spike_idx], spikes['clusters'][spike_idx], np.array([neuron]),
                                                      events, eventsStim, eventsMove,
                                                      align_epoch=[-0.2, 0.2], bin_size=fr_bin_size,
                                                      baseline_events=eventBase, base_epoch=base_epoch, smoothing=smoothing, norm=norm,
                                                      slide_kwargs=slide_kwargs_fr, kernel_kwargs=kernel_kwargs)
                #ax[1].set_xlim(left= -0.2 - fr_bin_size / 2, right= post_time_adjusted + fr_bin_size / 2) #Adjust later (MT)
            else:
                fr, fr_std, t = compute_psth(spikes['times'][spike_idx], spikes['clusters'][spike_idx], np.array([neuron]),
                                             events, align_epoch=[-0.2, 0.05], bin_size=fr_bin_size,
                                             baseline_events=eventBase, base_epoch=base_epoch, smoothing=smoothing, norm=norm,
                                             slide_kwargs=slide_kwargs_fr, kernel_kwargs=kernel_kwargs)
                #ax[1].set_xlim(left= -0.2 - fr_bin_size / 2, right= post_time_adjusted + fr_bin_size / 2) #Adjust later (MT)

            ax[1].plot(t, fr[0], c=str(1 - (base_grey + c * (1 - base_grey))))
            ax[1].fill_between(t, fr[0] + fr_std[0] / np.sqrt(len(events)), fr[0] - fr_std[0] / np.sqrt(len(events)),
                               color=str(1 - (base_grey + c * (1 - base_grey))), alpha=0.3)
            ax[1].set_ylim(-0.5, 20) #40) #, 24) #Adjust later (MT)
            ax[1].set_xlim(left= -0.2 - fr_bin_size / 2, right= post_time_adjusted + fr_bin_size / 2) #Adjust later (MT)

        else:
            fr, fr_std, t = compute_psth(spikes['times'][spike_idx], spikes['clusters'][spike_idx], np.array([neuron]),
                                         events, align_epoch=event_epoch, bin_size=fr_bin_size,
                                         baseline_events=eventBase, base_epoch=base_epoch, smoothing=smoothing, norm=norm,
                                         slide_kwargs=slide_kwargs_fr, kernel_kwargs=kernel_kwargs)
            ax[1].plot(t, fr[0], c=str(1 - (base_grey + c * (1 - base_grey))))
            ax[1].fill_between(t, fr[0] + fr_std[0] / np.sqrt(len(events)), fr[0] - fr_std[0] / np.sqrt(len(events)),
                               color=str(1 - (base_grey + c * (1 - base_grey))), alpha=0.3)
            ax[1].set_xlim(left=pre_time, right=post_time)

    ax[1].axvline(0, color=zero_line_c, ls='--')
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].set_ylabel("Firing rate (sp/s)")  # , size=labelsize + 3)
    ax[1].set_xlim(left=pre_time, right=post_time)
    #sns.despine(trim=True, ax=ax[1])
    # ax[1].tick_params(labelsize=labelsize)

    if plot_ff:
        for c in contrasts:
            events = eventTimes[trials['contrast'] == c]
            _, _, ff, t = compute_psth(spikes['times'][spike_idx], spikes['clusters'][spike_idx], np.array([neuron]),
                                       events, align_epoch=event_epoch, bin_size=ff_bin_size,
                                       baseline_events=eventBase, base_epoch=base_epoch, smoothing=smoothing, norm=norm,
                                       slide_kwargs=slide_kwargs_ff, return_ff=True)
            ax[2].plot(t, ff[0], c=str(1 - (base_grey + c * (1 - base_grey))))

        ax[2].axvline(0, color=zero_line_c, ls='--')
        ax[2].spines['right'].set_visible(False)
        ax[2].spines['top'].set_visible(False)
        ax[2].set_ylabel("Fano Factor")  # , size=labelsize + 3)
        ax[2].set_xlim(left=pre_time, right=post_time)
        # ax[2].tick_params(labelsize=labelsize)
        if align_event == 'move':
            ax[2].set_xlabel("Time from movement onset (s)")  # , size=labelsize + 3)
        else:
            ax[2].set_xlabel("Time from stimulus onset (s)")  # , size=labelsize + 3)
    else:
        if align_event == 'move':
            ax[1].set_xlabel("Time from movement onset (s)")  # , size=labelsize + 3)
        else:
            ax[1].set_xlabel("Time from stimulus onset (s)")  # , size=labelsize + 3)

    return ax

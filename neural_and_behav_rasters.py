from one.api import ONE
import numpy as np
import brainbox.behavior.wheel as wh
from brainbox.behavior.dlc import get_dlc_everything
from brainbox.task.trials import get_event_aligned_raster, filter_trials
import matplotlib.pyplot as plt


def plot_psth_raster(axs, t, psth, raster, title=None, ylabel=None, xlabel='Time', cmap='viridis',
                     clevels=None, tbin=1):
    ax = axs[0]
    psth_lines = []
    for ps in psth.keys():
        psth_lines.append(ax.plot(t, psth[ps]['vals']/tbin, **psth[ps]['linestyle'])[0])
        ax.fill_between(t, psth[ps]['vals'] / tbin + psth[ps]['err'] / tbin,
                        psth[ps]['vals'] / tbin - psth[ps]['err'] / tbin,
                        alpha=0.3, **psth[ps]['linestyle'])
    ax.legend(psth_lines, list(psth.keys()), fontsize='x-small')
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    if clevels is None:
        clevels = [np.nanmin(raster['vals'].ravel()), np.nanmax(raster['vals'].ravel())]
    ax = axs[1]
    ax.imshow(raster['vals'], aspect='auto', origin='lower',
              extent=np.r_[np.min(t), np.max(t), 0, raster['vals'].shape[0]], cmap=cmap,
              vmin=clevels[0], vmax=clevels[1])
    for d in raster['dividers']:
        ax.hlines(d, *ax.get_xlim(), color='k', linestyle='dashed')

    ax.vlines(0, *ax.get_ylim(), color='k', linestyle='dashed')

    ax.set_xlabel(xlabel)


def plot_neural_behav_raster(eid, probe, camera='left', clust_id=0, align_event='goCue_times',
                             order='trial num', sort='choice', one=None, spike_collection=None):
    """

    :param eid: session id
    :param probe: probe name
    :param camera: camera options are 'left' or 'right'
    :param clust_id: id of cluster to show
    :param align_event: event to align to, can find options in trials.keys() (any that have time)
    :param order: how to order trials, options are 'trial num' or 'reaction time'
    :param sort: how to differentiate trial, options are 'choice' (i.e. correct vs incorrect),
    'side' (i.e. left vs right), 'choice and side' (correct vs incorrect and left vs right)
    :return:
    """
    one = one or ONE()

    dlc = one.load_object(eid, f'{camera}Camera', collection='alf', attribute='dlc|times')
    dlc_all = get_dlc_everything(dlc, camera)
    wheel = one.load_object(eid, 'wheel', collection='alf',)
    trials = one.load_object(eid, 'trials')

    if spike_collection:
        collection = f'alf/{probe}/{spike_collection}'
    else:
        collection = f'alf/{probe}'

    spikes = one.load_object(eid, 'spikes', collection=collection, attribute='times|clusters')

    fig, axs = plt.subplots(2, 6, figsize=(18, 8), constrained_layout=True)
    # Spike
    tbin = 0.02
    spike_raster, t = get_event_aligned_raster(spikes.times[spikes.clusters == clust_id],
                                               trials[align_event], tbin=tbin)

    spike_raster_sorted, spike_psth = filter_trials(trials, spike_raster, align_event, order, sort)

    plot_psth_raster([axs[0, 0], axs[1, 0]], t, spike_psth, spike_raster_sorted,
                      title=f'Cluster {clust_id}', ylabel='Firing Rate', cmap='binary', tbin=tbin)

    # Wheel
    wheel_velocity = wh.velocity(wheel.timestamps, wheel.position)
    wheel_raster, t = get_event_aligned_raster(wheel.timestamps, trials[align_event],
                                                      values=wheel_velocity, tbin=tbin)
    wheel_raster_sorted, wheel_psth = filter_trials(trials, wheel_raster, align_event, order, sort)
    plot_psth_raster([axs[0, 1], axs[1, 1]], t, wheel_psth, wheel_raster_sorted,
                     title='wheel velocity', ylabel='rad/s', cmap='viridis')

    # Right paw
    paw_r_raster, t = get_event_aligned_raster(dlc_all.times, trials[align_event],
                                                values=dlc_all.dlc.paw_r_speed, tbin=tbin)
    paw_r_raster_sorted, paw_r_psth = filter_trials(trials, paw_r_raster, align_event, order, sort)
    plot_psth_raster([axs[0, 2], axs[1, 2]], t, paw_r_psth, paw_r_raster_sorted,
                     title='right paw speed', ylabel='px/s', cmap='viridis',
                     clevels=np.nanquantile(paw_r_raster_sorted['vals'], [0, 0.9]))

    # Left paw
    paw_l_raster, t = get_event_aligned_raster(dlc_all.times, trials[align_event],
                                                values=dlc_all.dlc.paw_l_speed, tbin=tbin)
    paw_l_raster_sorted, paw_l_psth = filter_trials(trials, paw_l_raster, align_event, order, sort)
    plot_psth_raster([axs[0, 3], axs[1, 3]], t, paw_l_psth, paw_l_raster_sorted,
                     title='left paw speed', ylabel='px/s', cmap='viridis',
                     clevels=np.nanquantile(paw_l_raster_sorted['vals'], [0, 0.9]))


    # Nose top
    nose_raster, t = get_event_aligned_raster(dlc_all.times, trials[align_event],
                                              values=dlc_all.dlc.nose_tip_speed, tbin=tbin)
    nose_raster_sorted, nose_psth = filter_trials(trials, nose_raster, align_event, order, sort)
    plot_psth_raster([axs[0, 4], axs[1, 4]], t, nose_psth, nose_raster_sorted,
                     title='nose tip speed', ylabel='px/s', cmap='viridis',
                     clevels=np.nanquantile(nose_raster_sorted['vals'], [0, 0.9]))

    # Licks
    lick_raster, t = get_event_aligned_raster(dlc_all.licks, trials[align_event], tbin=tbin)
    lick_raster_sorted, lick_psth = filter_trials(trials, lick_raster, align_event, order, sort)
    plot_psth_raster([axs[0, 5], axs[1, 5]], t, lick_psth, lick_raster_sorted,
                     title='licks', ylabel='px/s', cmap='binary')

    plt.suptitle(f'Align:{align_event} Ordered:{order} Camera:{camera}')

    plt.show()

    return fig



if __name__ == '__main__':

    eid = '56b57c38-2699-4091-90a8-aba35103155e'
    probe = 'probe01'

    # Example 1: left dlc camera, aligned to feedback_times, sorted by choice (correct incorrect),
    # cluster no. 378, using non default revision
    fig = plot_neural_behav_raster(eid, probe, clust_id=378, camera='left',
                                   align_event='feedback_times', sort='choice',
                                   spike_collection='ks2_preproc_tests')

    # Example 2: left dlc camera, aligned to goCue_times, sorted by choice and side (left right and
    # correct incorrect), order by reaction time
    fig = plot_neural_behav_raster(eid, probe, clust_id=378, camera='left',
                                   align_event='goCue_times', sort='choice and side',
                                   order='reaction time')

    # Example 3: right dlc camera, aligned to goCue_times, sorted by side (left right)
    # order by trial num
    fig = plot_neural_behav_raster(eid, probe, clust_id=378, camera='right',
                                   align_event='goCue_times', sort='side', order='trial num')


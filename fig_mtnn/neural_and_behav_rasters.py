from one.api import ONE
import numpy as np
import brainbox.behavior.wheel as wh
from brainbox.behavior.dlc import get_speed_for_features, get_licks, get_sniffs
from brainbox.task.trials import get_event_aligned_raster, filter_trials
import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd
from reproducible_ephys_functions import save_data_path

def plot_psth_raster(ax, t, psth, raster, title=None, ylabel=None, 
                     xlabel='time (sec)', 
                     stim_dir=None, cmap='viridis',
                     clevels=None, tbin=1):

    if clevels is None:
        clevels = [np.nanmin(raster['vals'].ravel()), np.nanmax(raster['vals'].ravel())]
     
    ax.set_title(title, fontdict={'fontsize': 27})
    ax.imshow(raster['vals'], aspect='auto', origin='lower',
              extent=np.r_[np.min(t), np.max(t), 0, raster['vals'].shape[0]], cmap=cmap,
              vmin=clevels[0], vmax=clevels[1], interpolation='none')
    for d in raster['dividers']:
        ax.hlines(d, *ax.get_xlim(), color='k', linestyle='dashed')

    ax.vlines(0, *ax.get_ylim(), color='k', linestyle='dashed')

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=26)
        ax.set_xticks(np.arange(-0.5,1.1,0.5))
        ax.set_xticklabels(np.arange(-0.5,1.1,0.5), fontsize=16)
    else:
        ax.set_xticks([])
        
    ax.set_yticks([])
        
    if stim_dir=="left":
        ax.set_ylabel('trials', fontsize=26)

def get_dlc_everything_skip_threshold(dlc_cam, camera):
    """
    Get out features of interest for dlc
    :param dlc_cam: dlc object
    :param camera: camera type e.g 'left', 'right'
    :return:
    """

    aligned = True
    if dlc_cam.times.shape[0] != dlc_cam.dlc.shape[0]:
        # logger warning and print out status of the qc, specific serializer django!
        logger.warning('Dimension mismatch between dlc points and timestamps')
        min_samps = min(dlc_cam.times.shape[0], dlc_cam.dlc.shape[0])
        dlc_cam.times = dlc_cam.times[:min_samps]
        dlc_cam.dlc = dlc_cam.dlc[:min_samps]
        aligned = False

    dlc_cam.dlc = get_speed_for_features(dlc_cam.dlc, dlc_cam.times, camera)
    dlc_cam['licks'] = get_licks(dlc_cam.dlc, dlc_cam.times)
    dlc_cam['sniffs'] = get_sniffs(dlc_cam.dlc, dlc_cam.times)
    dlc_cam['aligned'] = aligned

    return dlc_cam

def normalize(raster):
    r = raster['vals']
    raster['vals'] = 2*(r-r.min())/(r.max()-r.min())-1
    return raster

def plot_neural_behav_raster(eid, probe, trial_idx, stim_dir='left', camera='left',
                             axs=None, fig=None, epoch=[-0.5,1.0],
                             clust_id=0, align_event='firstMovement_times',
                             order='reaction time', sort='choice and side', one=None, 
                             spike_collection='pykilosort', tbin=0.05):
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

    dlc = one.load_object(eid, f'{camera}Camera', collection='alf', attribute=['dlc', 'times'])
    # override with lpks
    lpks = pd.read_parquet(save_data_path(figure='fig_mtnn').joinpath('lpks', f'{eid}._ibl_leftCamera.pose.pqt'))
    dlc['dlc']['paw_l_x'] = lpks['paw_l_x']
    dlc['dlc']['paw_l_y'] = lpks['paw_l_y']
    dlc['dlc']['paw_r_x'] = lpks['paw_r_x']
    dlc['dlc']['paw_r_y'] = lpks['paw_r_y']
    
    dlc['dlc']['pupil_top_r_x'] = lpks['pupil_top_r_x']
    dlc['dlc']['pupil_top_r_y'] = lpks['pupil_top_r_y']
    dlc['dlc']['pupil_bottom_r_x'] = lpks['pupil_bottom_r_x']
    dlc['dlc']['pupil_bottom_r_y'] = lpks['pupil_bottom_r_y']
    dlc['dlc']['pupil_left_r_x'] = lpks['pupil_left_r_x']
    dlc['dlc']['pupil_left_r_y'] = lpks['pupil_left_r_y']
    dlc['dlc']['pupil_right_r_x'] = lpks['pupil_right_r_x']
    dlc['dlc']['pupil_right_r_y'] = lpks['pupil_right_r_y']
    
    dlc_all = get_dlc_everything_skip_threshold(dlc, camera)
    wheel = one.load_object(eid, 'wheel', collection='alf',)
    me = one.load_dataset(eid, 'leftCamera.ROIMotionEnergy.npy')
    trials = one.load_object(eid, 'trials')
    trials_all = deepcopy(trials)
    if trial_idx is not None:
        for key in trials.keys():
            trials[key] = trials[key][trial_idx]

    if spike_collection:
        collection = f'alf/{probe}/{spike_collection}'
    else:
        collection = f'alf/{probe}'

    spikes = one.load_object(eid, 'spikes', collection=collection, attribute=['times','clusters'],
                             revision='2024-03-22')

    if axs is None:
        fig, axs = plt.subplots(2, 6, figsize=(18, 8), constrained_layout=True)
    # Spike
    spike_raster, t = get_event_aligned_raster(spikes.times[spikes.clusters == clust_id],
                                               trials[align_event], tbin=tbin, epoch=epoch)
    spike_raster_sorted, spike_psth = filter_trials(trials, spike_raster, align_event, 
                                                    order=order, sort=sort)
    spike_raster_all, _ = get_event_aligned_raster(spikes.times[spikes.clusters == clust_id],
                                               trials_all[align_event], tbin=tbin, epoch=epoch)
    spike_raster_all_sorted, _ = filter_trials(trials_all, spike_raster_all, align_event, 
                                                    order=order, sort=sort)
    spike_raster_clevels = np.percentile(spike_raster_all_sorted['vals'], [0.01,99.9])
#     spike_raster_sorted = normalize(spike_raster_sorted)
    plot_psth_raster(axs[0][0], t, spike_psth, spike_raster_sorted,
                     title='firing rate', 
                     ylabel=None, stim_dir=stim_dir,
                     cmap='binary', tbin=tbin, xlabel=None, clevels=spike_raster_clevels)

    # Wheel
    wheel_velocity = wh.velocity(wheel.timestamps, wheel.position)
    wheel_raster, t = get_event_aligned_raster(wheel.timestamps, trials[align_event],
                                               values=wheel_velocity, tbin=tbin, epoch=epoch)
    wheel_raster_sorted, wheel_psth = filter_trials(trials, wheel_raster, align_event, 
                                                    order=order, sort=sort)
    wheel_raster_all, _ = get_event_aligned_raster(wheel.timestamps, trials_all[align_event],
                                               values=wheel_velocity, tbin=tbin, epoch=epoch)
    wheel_raster_all_sorted, _ = filter_trials(trials_all, wheel_raster_all, align_event, 
                                                    order=order, sort=sort)
#     wheel_raster_sorted = normalize(wheel_raster_sorted)
    wheel_clevels = np.percentile(wheel_raster_all_sorted['vals'], [0.1,99.8])
    plot_psth_raster(axs[0][1], t, wheel_psth, wheel_raster_sorted,
                     title=f'behavioral variable\nraster plots ({stim_dir} stimulus)\n\nwheel velocity',
                     ylabel='rad/s', cmap='binary', xlabel=None, clevels=wheel_clevels)

    # Right paw
    paw_r_raster, t = get_event_aligned_raster(dlc_all.times, trials[align_event],
                                                values=dlc_all.dlc.paw_r_speed, tbin=tbin, epoch=epoch)
    paw_r_raster_sorted, paw_r_psth = filter_trials(trials, paw_r_raster, align_event,
                                                    order=order, sort=sort)
    paw_r_raster_all, _ = get_event_aligned_raster(dlc_all.times, trials_all[align_event],
                                               values=dlc_all.dlc.paw_r_speed, tbin=tbin, epoch=epoch)
    paw_r_raster_all_sorted, _ = filter_trials(trials_all, paw_r_raster_all, align_event, 
                                                    order=order, sort=sort)
    paw_r_clevels = np.percentile(paw_r_raster_all_sorted['vals'], [0.01,99.9])
#     paw_r_raster_sorted = normalize(paw_r_raster_sorted)
    plot_psth_raster(axs[0][2], t, paw_r_psth, paw_r_raster_sorted,
                     title='right paw speed', ylabel='px/s', cmap='binary', 
                     xlabel=None, clevels=paw_r_clevels)
    
    # Motion energy
    me_raster, t = get_event_aligned_raster(dlc_all.times, trials[align_event],
                                                values=me, tbin=tbin, epoch=epoch)
    me_raster_sorted, me_psth = filter_trials(trials, me_raster, align_event, 
                                              order=order, sort=sort)
    me_raster_all, _ = get_event_aligned_raster(dlc_all.times, trials_all[align_event],
                                               values=me, tbin=tbin, epoch=epoch)
    me_raster_all_sorted, _ = filter_trials(trials_all, me_raster_all, align_event, 
                                                    order=order, sort=sort)
    me_clevels = np.percentile(me_raster_all_sorted['vals'], [0.01,99.9])
#     me_raster_sorted = normalize(me_raster_sorted)
    plot_psth_raster(axs[1][0], t, me_psth, me_raster_sorted,
                     title='motion energy', ylabel=None, cmap='binary', 
                     stim_dir=stim_dir, clevels=me_clevels)

    # Nose top
    nose_raster, t = get_event_aligned_raster(dlc_all.times, trials[align_event],
                                              values=dlc_all.dlc.nose_tip_speed, tbin=tbin, epoch=epoch)
    nose_raster_sorted, nose_psth = filter_trials(trials, nose_raster, align_event, 
                                                  order=order, sort=sort)
    nose_raster_all, _ = get_event_aligned_raster(dlc_all.times, trials_all[align_event],
                                               values=dlc_all.dlc.nose_tip_speed, tbin=tbin, epoch=epoch)
    nose_raster_all_sorted, _ = filter_trials(trials_all, nose_raster_all, align_event, 
                                                    order=order, sort=sort)
    nose_clevels = np.percentile(nose_raster_all_sorted['vals'], [0.01,99.9])
#     nose_raster_sorted = normalize(nose_raster_sorted)
    plot_psth_raster(axs[1][1], t, nose_psth, nose_raster_sorted,
                     title='nose tip speed', ylabel='px/s', cmap='binary',
                     clevels=nose_clevels)

    # Licks
    lick_raster, t = get_event_aligned_raster(dlc_all.licks, trials[align_event], tbin=tbin, epoch=epoch)
    lick_raster_sorted, lick_psth = filter_trials(trials, lick_raster, align_event,
                                                  order=order, sort=sort)
    lick_raster_all, _ = get_event_aligned_raster(dlc_all.licks, trials_all[align_event],
                                                  tbin=tbin, epoch=epoch)
    lick_raster_all_sorted, _ = filter_trials(trials_all, lick_raster_all, align_event, 
                                                    order=order, sort=sort)
#     lick_clevels = np.percentile(lick_raster_all_sorted['vals'], [0.01,99.9])
#     lick_raster_sorted = normalize(lick_raster_sorted)
    plot_psth_raster(axs[1][2], t, lick_psth, lick_raster_sorted,
                     title='licks', ylabel='px/s', cmap='binary')#, clevels=lick_clevels)
    
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


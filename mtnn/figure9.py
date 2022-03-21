import numpy as np
from collections import defaultdict

import figrid as fg
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pylab as pl
from tqdm import notebook

import sys, os
sys.path.append('..')
from neural_and_behav_rasters import plot_neural_behav_raster

from utils import *

def split_by_stimulus(feature):
    
    stim = feature[0,:,:,cov_idx_dict['stimuli'][0]:cov_idx_dict['stimuli'][1]].sum(1)
    left_idx = np.nonzero(stim[:,0])
    right_idx = np.nonzero(stim[:,1])
    
    contrast_left = np.zeros(feature.shape[1]).astype(bool)
    contrast_left[left_idx] = True
                   
    contrast_right = np.zeros(feature.shape[1]).astype(bool)
    contrast_right[right_idx] = True
    
    return contrast_left, contrast_right

def split_by_feedback(feature):
    
    feedback = feature[0,:,:,cov_idx_dict['reward'][0]:cov_idx_dict['reward'][1]].sum(1)
    incorrect_idx = np.nonzero(feedback[:,0])
    correct_idx = np.nonzero(feedback[:,1])
    
    correct = np.zeros(feature.shape[1]).astype(bool)
    correct[correct_idx] = True
                   
    incorrect = np.zeros(feature.shape[1]).astype(bool)
    incorrect[incorrect_idx] = True
    
    return correct, incorrect

def order_by_response_time(feature):
    
    response = feature[0,:,:,cov_idx_dict['choice'][0]:cov_idx_dict['choice'][1]].sum(-1)
    response_times = np.nonzero(response)[1]
    ordering = np.argsort(response_times)[::-1]
    
    return ordering

def split_reorder_data(feature, pred, obs, trial):
    
    # split by stimulus
    left_stim_bool, right_stim_bool = split_by_stimulus(feature)
    left_feature, right_feature = feature[:,left_stim_bool], feature[:,right_stim_bool]
    left_pred, right_pred = pred[:,left_stim_bool], pred[:,right_stim_bool]
    left_obs, right_obs = obs[:,left_stim_bool], obs[:,right_stim_bool]
    left_trial, right_trial = trial[left_stim_bool], trial[right_stim_bool]

    # split by correct/incorrect
    left_o_bool, left_x_bool = split_by_feedback(left_feature)
    left_correct_feature, left_incorrect_feature = left_feature[:,left_o_bool], left_feature[:,left_x_bool]
    left_correct_pred, left_incorrect_pred = left_pred[:,left_o_bool], left_pred[:,left_x_bool]
    left_correct_obs, left_incorrect_obs = left_obs[:,left_o_bool], left_obs[:,left_x_bool]

    right_o_bool, right_x_bool = split_by_feedback(right_feature)
    right_correct_feature, right_incorrect_feature = right_feature[:,right_o_bool], right_feature[:,right_x_bool]
    right_correct_pred, right_incorrect_pred = right_pred[:,right_o_bool], right_pred[:,right_x_bool]
    right_correct_obs, right_incorrect_obs = right_obs[:,right_o_bool], right_obs[:,right_x_bool]

    # order by response time
    left_o_order = order_by_response_time(left_correct_feature)
    left_correct_feature = left_correct_feature[:,left_o_order]
    left_correct_pred = left_correct_pred[:,left_o_order]
    left_correct_obs = left_correct_obs[:,left_o_order]

    left_x_order = order_by_response_time(left_incorrect_feature)
    left_incorrect_feature = left_incorrect_feature[:,left_x_order]
    left_incorrect_pred = left_incorrect_pred[:,left_x_order]
    left_incorrect_obs = left_incorrect_obs[:,left_x_order]
    n_left_incorrect = left_incorrect_feature.shape[1]

    right_o_order = order_by_response_time(right_correct_feature)
    right_correct_feature = right_correct_feature[:,right_o_order]
    right_correct_pred = right_correct_pred[:,right_o_order]
    right_correct_obs = right_correct_obs[:,right_o_order]

    right_x_order = order_by_response_time(right_incorrect_feature)
    right_incorrect_feature = right_incorrect_feature[:,right_x_order]
    right_incorrect_pred = right_incorrect_pred[:,right_x_order]
    right_incorrect_obs = right_incorrect_obs[:,right_x_order]
    n_right_incorrect = right_incorrect_feature.shape[1]

    # stack left/right
    if left_incorrect_feature.shape[1] != 0:
        left_feature_concat = np.concatenate((left_incorrect_feature, left_correct_feature), axis=1)
        left_pred_concat = np.concatenate((left_incorrect_pred, left_correct_pred), axis=1)
        left_obs_concat = np.concatenate((left_incorrect_obs, left_correct_obs), axis=1)
    else:
        left_feature_concat = left_correct_feature
        left_pred_concat = left_correct_pred
        left_obs_concat = left_correct_obs
        
    if right_incorrect_feature.shape[1] != 0:
        right_feature_concat = np.concatenate((right_incorrect_feature, right_correct_feature), axis=1)
        right_pred_concat = np.concatenate((right_incorrect_pred, right_correct_pred), axis=1)
        right_obs_concat = np.concatenate((right_incorrect_obs, right_correct_obs), axis=1)
    else:
        right_feature_concat = right_correct_feature
        right_pred_concat = right_correct_pred
        right_obs_concat = right_correct_obs
    
    left = (left_feature_concat,left_pred_concat,
            left_obs_concat,left_trial.astype(int),n_left_incorrect)
    right = (right_feature_concat,right_pred_concat,
             right_obs_concat,right_trial.astype(int),n_right_incorrect)
    
    return left, right

def get_ticks(feature):
    
    ticks = defaultdict(list)
    for i in range(feature.shape[1]):
        tr = feature[0,i]
        
        # stim onset tick
        stim = tr[:,cov_idx_dict['stimuli'][0]:cov_idx_dict['stimuli'][1]].sum(1)
        stimOnTick = np.nonzero(stim)[0][0]
        ticks['stimOn'].append(stimOnTick)
        
        # feedback tick
        feedback = tr[:,cov_idx_dict['reward'][0]:cov_idx_dict['reward'][1]].sum(1)
        feedbackTick = np.nonzero(feedback)[0][0]
        ticks['feedback'].append(feedbackTick)
    
    return ticks

def make_fig_ax():
    
    xsplit = ([0,0.46], [0.51,0.97], [0.98,1])
    ysplit = ([0.05,0.22], [0.23,0.40], [0.41,0.58], [0.73, 1], [0.23, 0.58])
    
    fig = plt.figure(figsize=(30,20))
    ax = {'panel_A': fg.place_axes_on_grid(fig, xspan=xsplit[0], yspan=ysplit[0]),
          'panel_B': fg.place_axes_on_grid(fig, xspan=xsplit[1], yspan=ysplit[0]),
          'panel_C': fg.place_axes_on_grid(fig, xspan=xsplit[0], yspan=ysplit[1]),
          'panel_D': fg.place_axes_on_grid(fig, xspan=xsplit[1], yspan=ysplit[1]),
          'panel_E': fg.place_axes_on_grid(fig, xspan=xsplit[0], yspan=ysplit[2]),
          'panel_F': fg.place_axes_on_grid(fig, xspan=xsplit[1], yspan=ysplit[2]),
          'panel_G': fg.place_axes_on_grid(fig, xspan=xsplit[0], yspan=ysplit[3], 
                                           dim=[2, 3], hspace=0.25),
          'panel_H': fg.place_axes_on_grid(fig, xspan=xsplit[1], yspan=ysplit[3], 
                                           dim=[2, 3], hspace=0.25),
          'panel_I': fg.place_axes_on_grid(fig, xspan=xsplit[2], yspan=ysplit[4])}
    
    # Add subplot labels
    labels = [{'label_text':'a', 'xpos':0, 'ypos':0.02, 'fontsize':30, 'weight': 'bold',
               'ha': 'right', 'va': 'bottom'},
              {'label_text':'b', 'xpos':0, 'ypos':0.65, 'fontsize':30, 'weight': 'bold',
               'ha': 'right', 'va': 'bottom'}]
    
    return fig, ax, labels

def figure_style(return_colors=False):
    """
    Set seaborn style for plotting figures
    """
    sns.set(style="ticks", context="paper",
            font="DejaVu Sans",
            rc={"font.size": 24,
                "axes.titlesize": 24,
                "axes.labelsize": 24,
                "axes.linewidth": 0.5,
                "lines.linewidth": 1,
                "lines.markersize": 4,
                "xtick.labelsize": 16,
                "ytick.labelsize": 16,
                "savefig.transparent": True,
                "xtick.major.size": 2.5,
                "ytick.major.size": 2.5,
                "xtick.major.width": 0.5,
                "ytick.major.width": 0.5,
                "xtick.minor.size": 2,
                "ytick.minor.size": 2,
                "xtick.minor.width": 0.5,
                "ytick.minor.width": 0.5
                })
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    if return_colors:
        return {'PPC': sns.color_palette('colorblind')[0],
                'CA1': sns.color_palette('colorblind')[2],
                'DG': sns.color_palette('muted')[2],
                'LP': sns.color_palette('colorblind')[4],
                'PO': sns.color_palette('colorblind')[6],
                'RS': sns.color_palette('Set2')[0],
                'FS': sns.color_palette('Set2')[1],
                'RS1': sns.color_palette('Set2')[2],
                'RS2': sns.color_palette('Set2')[3]}

def generate_figure_9(feature_list, pred_list, obs_list, neu_list, sess_list, trial_list,
                      bin_size=0.05, which_sess=None, savefig=False, plot_subsample_ratio=0.2):
    '''
    which_sess: list
    '''
    
    if which_sess is None:
        which_sess = range(len(sess_list))
        
    figure_style()
    acronym_dict_reverse = get_acronym_dict_reverse()
        
    for i in which_sess:
        sess_info = sess_list[i]
        eid = sess_info['session']['id']
        probe = sess_info['probe_name']
        
        feature = feature_list[i]
        pred = pred_list[i]
        obs = obs_list[i]
        neu = neu_list[i]
        trial = trial_list[i]
        
        left, right = split_reorder_data(feature, pred, obs, trial)
        left_ticks, right_ticks= get_ticks(left[0]), get_ticks(right[0])
        left_trial_idx, right_trial_idx = left[3], right[3]
        
        if plot_subsample_ratio < 1.0:
            n_neurons = neu.shape[0]
            n_samples = int(n_neurons*plot_subsample_ratio)
            selected_neurons = np.random.choice(np.arange(n_neurons), size=n_samples, replace=False)
        else:
            selected_neurons = np.arange(n_neurons)

        for j, neuron in notebook.tqdm(enumerate(neu)):
            if j not in selected_neurons:
                continue
            
            left_pred_j = left[1][j]
            left_obs_j = left[2][j]
            right_pred_j = right[1][j]
            right_obs_j = right[2][j]

            region_idx = np.nonzero(left[0][j,0,0,acronym_offset:noise_offset])[0][0]
            region = acronym_dict_reverse[region_idx]
            
            max_fr = max([left_pred_j.max(), left_obs_j.max(), 
                          right_pred_j.max(), right_obs_j.max()])
            max_psth = max([left_pred_j.mean(0).max(), left_obs_j.mean(0).max(), 
                            right_pred_j.mean(0).max(), right_obs_j.mean(0).max()])
            
            fig, ax, labels = make_fig_ax()

            ax['panel_A'].plot(left_pred_j.mean(0), color='r', lw=3, label='predicted')
            ax['panel_A'].plot(left_obs_j.mean(0), color='k', lw=3, label='observed')
            ax['panel_A'].set_ylim(0, max_psth*1.2)
            ax['panel_A'].set_xlim(-0.5,pred.shape[-1]-0.5)
            ax['panel_A'].axvline(10.5, color='k', linestyle='--')
            ax['panel_A'].set_xticks([])
            ax['panel_A'].set_title('left stimulus')
            ax['panel_A'].set_ylabel('firing rate (Hz)')
            ax['panel_A'].legend(fontsize=20)
            
            ax['panel_B'].plot(right_pred_j.mean(0), color='r', lw=3, label='predicted')
            ax['panel_B'].plot(right_obs_j.mean(0), color='k', lw=3, label='observed')
            ax['panel_B'].set_ylim(0, max_psth*1.2)
            ax['panel_B'].set_xlim(-0.5,pred.shape[-1]-0.5)
            ax['panel_B'].axvline(10.5, color='k', linestyle='--')
            ax['panel_B'].set_yticks([])
            ax['panel_B'].set_xticks([])
            ax['panel_B'].set_title('right stimulus')
            ax['panel_B'].legend(fontsize=20)
            
            ax['panel_C'].imshow(left_obs_j, aspect='auto', vmin=0, vmax=max_fr, 
                                 cmap=plt.get_cmap('binary'), interpolation='none')
            if left[-1] > 0:
                ax['panel_C'].axhline(left[-1]-0.5, color='k', linestyle='--')
            ax['panel_C'].axvline(10.5, color='k', linestyle='--')
            ax['panel_C'].set_xticks([])
            ax['panel_C'].set_yticks([])
            ax['panel_C'].set_ylabel('observed\nraster plot\ntrials')
            for k in range(left_trial_idx.shape[0]):
                ax['panel_C'].plot([left_ticks['stimOn'][k]+0.5,left_ticks['stimOn'][k]+0.5],
                                   [k-0.5,k+0.5], color='b', lw=2)
                ax['panel_C'].plot([left_ticks['feedback'][k]+0.5,left_ticks['feedback'][k]+0.5],
                                   [k-0.5,k+0.5], color='g', lw=2)
            
            ax['panel_D'].imshow(right_obs_j, aspect='auto', vmin=0, vmax=max_fr,
                                 cmap=plt.get_cmap('binary'), interpolation='none')
            if right[-1] > 0:
                ax['panel_D'].axhline(right[-1]-0.5, color='k', linestyle='--')
            ax['panel_D'].axvline(10.5, color='k', linestyle='--')
            ax['panel_D'].set_yticks([])
            ax['panel_D'].set_xticks([])
            for k in range(right_trial_idx.shape[0]):
                ax['panel_D'].plot([right_ticks['stimOn'][k]+0.5,right_ticks['stimOn'][k]+0.5],
                                   [k-0.5,k+0.5], color='b', lw=2)
                ax['panel_D'].plot([right_ticks['feedback'][k]+0.5,right_ticks['feedback'][k]+0.5],
                                   [k-0.5,k+0.5], color='g', lw=2)
            
            ax['panel_E'].imshow(left_pred_j, aspect='auto', vmin=0, vmax=max_fr, 
                                 cmap=plt.get_cmap('binary'), interpolation='none')
            if left[-1] > 0:
                ax['panel_E'].axhline(left[-1]-0.5, color='k', linestyle='--')
            ax['panel_E'].axvline(10.5, color='k', linestyle='--')
            ax['panel_E'].set_xlabel('time (sec)')
            ax['panel_E'].set_yticks([])
            ax['panel_E'].set_ylabel('predicted\nraster plot\ntrials')
            ax['panel_E'].set_xticks([0.5, 10.5, 20.5, 29.5])
            ax['panel_E'].set_xticklabels(np.arange(-0.5,1.1,0.5))
            for k in range(left_trial_idx.shape[0]):
                ax['panel_E'].plot([left_ticks['stimOn'][k]+0.5,left_ticks['stimOn'][k]+0.5],
                                   [k-0.5,k+0.5], color='b', lw=2)
                ax['panel_E'].plot([left_ticks['feedback'][k]+0.5,left_ticks['feedback'][k]+0.5],
                                   [k-0.5,k+0.5], color='g', lw=2)
            
            ax['panel_F'].imshow(right_pred_j, aspect='auto', vmin=0, vmax=max_fr, 
                                 cmap=plt.get_cmap('binary'), interpolation='none')
            if right[-1] > 0:
                ax['panel_F'].axhline(right[-1]-0.5, color='k', linestyle='--')
            ax['panel_F'].axvline(10.5, color='k', linestyle='--')
            ax['panel_F'].set_yticks([])
            ax['panel_F'].set_xlabel('time (sec)')
            ax['panel_F'].set_xticks([0.5, 10.5, 20.5, 29.5])
            ax['panel_F'].set_xticklabels(np.arange(-0.5,1.1,0.5))
            for k in range(right_trial_idx.shape[0]):
                ax['panel_F'].plot([right_ticks['stimOn'][k]+0.5,right_ticks['stimOn'][k]+0.5],
                                   [k-0.5,k+0.5], color='b', lw=2, 
                                   label='stim onset' if k==0 else None)
                ax['panel_F'].plot([right_ticks['feedback'][k]+0.5,right_ticks['feedback'][k]+0.5],
                                   [k-0.5,k+0.5], color='g', lw=2, 
                                   label='feedback' if k==0 else None)
            ax['panel_F'].legend(fontsize=20)

            plot_neural_behav_raster(eid, probe, trial_idx=left_trial_idx,
                                     fig=fig, axs=ax['panel_G'], clust_id=neuron, stim_dir='left')
            
            plot_neural_behav_raster(eid, probe, trial_idx=right_trial_idx,
                                     fig=fig, axs=ax['panel_H'], clust_id=neuron, stim_dir='right')
            
            img = ax['panel_I'].imshow(left_obs_j, vmin=0, vmax=max_fr, 
                                 aspect='auto', cmap=plt.get_cmap('binary'))
            pl.gca().set_visible(False)
            divider = make_axes_locatable(ax['panel_I'])
            colorbar_axes = divider.append_axes("right", size="100%", pad=0.1)
            cbar = fig.colorbar(img, cax=colorbar_axes, orientation='vertical')
            cbar.ax.tick_params(labelsize=18)
            cbar.ax.set_xlabel('spikes/sec', fontsize=20, labelpad=32.0)

            fg.add_labels(fig, labels)
            
            if savefig:
                savedir = f'plots/figure9/{eid}'
                if not os.path.exists(savedir):
                    os.makedirs(savedir)
                figname = os.path.join(savedir, f'{region}_id={neuron}.png')
                plt.savefig(figname,bbox_inches='tight', facecolor='white')
            
            plt.show()
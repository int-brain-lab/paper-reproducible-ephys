import numpy as np
from collections import defaultdict
from tqdm import notebook

import figrid as fg
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pylab as pl
from tqdm import notebook
from sklearn.metrics import r2_score

from reproducible_ephys_functions import save_data_path, save_figure_path, figure_style, LAB_MAP
from fig_mtnn.neural_and_behav_rasters import plot_neural_behav_raster
from fig_mtnn.utils import (cov_idx_dict, get_acronym_dict_reverse, compute_mean_frs, acronym_offset, noise_offset, lab_offset,
                                         session_offset, xyz_offset, stimulus_offset, leave_out_covs_for_glm, reshape_flattened)
from fig_mtnn.mtnn import load_test_model

data_load_path = save_data_path(figure='fig_mtnn').joinpath('mtnn_data')
save_path = save_figure_path(figure='fig_mtnn')

rng = np.random.default_rng(seed=0b01101001 + 0b01100010 + 0b01101100)

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
    
    xsplit = ([0,0.475], [0.495,0.97], [0.975,1])
    ysplit = ([0.05,0.22], [0.23,0.40], [0.41,0.58], [0.745, 1], [0.23, 0.58])
    
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
    labels = [{'label_text':'a', 'xpos':0, 'ypos':0.02, 'fontsize':45, 'weight': 'bold',
               'ha': 'right', 'va': 'bottom'},
              {'label_text':'b', 'xpos':0, 'ypos':0.65, 'fontsize':45, 'weight': 'bold',
               'ha': 'right', 'va': 'bottom'}]
    
    return fig, ax, labels


def generate_figure_9(feature_list, pred_list, obs_list, neu_list, sess_list, trial_list,
                      bin_size=0.05, which_sess=None, savefig=False, 
                      plot_subsample_ratio=0.2, plot_neurons = None, fr_upper_threshold=None):
    '''
    which_sess: list
    '''
    
    if which_sess is None:
        which_sess = range(len(sess_list))
        
    figure_style()
    acronym_dict_reverse = get_acronym_dict_reverse()
        
    for i in which_sess:
        sess_info = sess_list[i]#.tolist()
        eid = sess_info['session']['id']
        subject = sess_info['session']['subject']
        probe = sess_info['probe_name']
        
        feature = feature_list[i]
        pred = pred_list[i]
        obs = obs_list[i]
        neu = neu_list[i]
        trial = trial_list[i]
        
        left, right = split_reorder_data(feature, pred, obs, trial)
        left_ticks, right_ticks= get_ticks(left[0]), get_ticks(right[0])
        left_trial_idx, right_trial_idx = left[3], right[3]
        
        n_neurons = neu.shape[0]
        if plot_subsample_ratio < 1.0:
            n_samples = int(n_neurons*plot_subsample_ratio)
            selected_neurons = rng.choice(np.arange(n_neurons), size=n_samples, replace=False)
        else:
            selected_neurons = np.arange(n_neurons)
            
        if plot_neurons is not None:
            selected_neurons = selected_neurons[plot_neurons]

        for j, neuron in notebook.tqdm(enumerate(neu)):
            if j not in selected_neurons:
                continue
            
            left_pred_j = left[1][j]
            left_obs_j = left[2][j]
            right_pred_j = right[1][j]
            right_obs_j = right[2][j]

            region_idx = np.nonzero(left[0][j,0,0,acronym_offset:noise_offset])[0][0]
            region = acronym_dict_reverse[region_idx]
            
            if fr_upper_threshold is not None:
                mean_fr_j = np.concatenate([left_obs_j,right_obs_j], axis=0).mean()
                if mean_fr_j >= fr_upper_threshold:
                    continue
                else:
                    print(f'mean fr: {mean_fr_j}')
            
            r2_psth = r2_score(np.concatenate([left_obs_j,right_obs_j], axis=0).mean(0), 
                           np.concatenate([left_pred_j,right_pred_j], axis=0).mean(0), 
                           multioutput='raw_values')[0]
            r2 = r2_score(np.concatenate([left_obs_j,right_obs_j], axis=0).flatten(), 
                           np.concatenate([left_pred_j,right_pred_j], axis=0).flatten(), 
                           multioutput='raw_values')[0]
            print(f'eid: {eid}, brain region: {region}, neuron id: {neuron}')
            print('R2: {:.3f}, R2 on PETH: {:.3f}\nsubject={}, region={}'.format(r2, r2_psth, subject, region))
            
            max_fr = max([left_pred_j.max(), left_obs_j.max(), 
                          right_pred_j.max(), right_obs_j.max()])
            max_psth = max([left_pred_j.mean(0).max(), left_obs_j.mean(0).max(), 
                            right_pred_j.mean(0).max(), right_obs_j.mean(0).max()])
            
            fig, ax, labels = make_fig_ax()

            ax['panel_A'].plot(left_pred_j.mean(0), color='r', lw=3, label='predicted')
            ax['panel_A'].plot(left_obs_j.mean(0), color='k', lw=3, label='observed')
            ax['panel_A'].set_ylim(0, max_psth*1.2)
            ax['panel_A'].set_xlim(-0.5,pred.shape[-1]-0.5)
            ax['panel_A'].axvline(10.5, color='k', linestyle='--', lw=3)
            ax['panel_A'].set_xticks([])
            ax['panel_A'].set_title('left stimulus', fontsize=28)
            ax['panel_A'].set_ylabel('firing rate (Hz)', fontsize=26)
            ax['panel_A'].set_yticks(np.arange(0,max_psth*1.2,20).astype(int))
            ax['panel_A'].set_yticklabels(np.arange(0,max_psth*1.2,20).astype(int), fontsize=22)
            ax['panel_A'].legend(fontsize=26)

            ax['panel_B'].plot(right_pred_j.mean(0), color='r', lw=3, label='predicted')
            ax['panel_B'].plot(right_obs_j.mean(0), color='k', lw=3, label='observed')
            ax['panel_B'].set_ylim(0, max_psth*1.2)
            ax['panel_B'].set_xlim(-0.5,pred.shape[-1]-0.5)
            ax['panel_B'].axvline(10.5, color='k', linestyle='--', lw=3)
            ax['panel_B'].set_yticks([])
            ax['panel_B'].set_xticks([])
            ax['panel_B'].set_title('right stimulus', fontsize=28)
            ax['panel_B'].legend(fontsize=26)

            ax['panel_C'].imshow(left_obs_j, aspect='auto', vmin=0, vmax=max_fr, 
                                 cmap=plt.get_cmap('binary'), interpolation='none')
            if left[-1] > 0:
                ax['panel_C'].axhline(left[-1]-0.5, color='k', linestyle='--', lw=3)
            ax['panel_C'].axvline(10.5, color='k', linestyle='--', lw=3)
            ax['panel_C'].set_xticks([])
            ax['panel_C'].set_yticks([])
            ax['panel_C'].set_ylabel('observed\nraster plot\ntrials', fontsize=26)
            for k in range(left_trial_idx.shape[0]):
                ax['panel_C'].plot([left_ticks['stimOn'][k]+0.5,left_ticks['stimOn'][k]+0.5],
                                   [k-0.5,k+0.5], color='b', lw=2)
                ax['panel_C'].plot([left_ticks['feedback'][k]+0.5,left_ticks['feedback'][k]+0.5],
                                   [k-0.5,k+0.5], color='g', lw=2)
            
            ax['panel_D'].imshow(right_obs_j, aspect='auto', vmin=0, vmax=max_fr,
                                 cmap=plt.get_cmap('binary'), interpolation='none')
            if right[-1] > 0:
                ax['panel_D'].axhline(right[-1]-0.5, color='k', linestyle='--', lw=3)
            ax['panel_D'].axvline(10.5, color='k', linestyle='--', lw=3)
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
                ax['panel_E'].axhline(left[-1]-0.5, color='k', linestyle='--', lw=3)
            ax['panel_E'].axvline(10.5, color='k', linestyle='--', lw=3)
            ax['panel_E'].set_xlabel('time (sec)', fontsize=26)
            ax['panel_E'].set_yticks([])
            ax['panel_E'].set_ylabel('predicted\nraster plot\ntrials', fontsize=26)
            ax['panel_E'].set_xticks([0.5, 10.5, 20.5, 29.5])
            ax['panel_E'].set_xticklabels(np.arange(-0.5,1.1,0.5), fontsize=22)
            for k in range(left_trial_idx.shape[0]):
                ax['panel_E'].plot([left_ticks['stimOn'][k]+0.5,left_ticks['stimOn'][k]+0.5],
                                   [k-0.5,k+0.5], color='b', lw=3)
                ax['panel_E'].plot([left_ticks['feedback'][k]+0.5,left_ticks['feedback'][k]+0.5],
                                   [k-0.5,k+0.5], color='g', lw=3)
            
            ax['panel_F'].imshow(right_pred_j, aspect='auto', vmin=0, vmax=max_fr, 
                                 cmap=plt.get_cmap('binary'), interpolation='none')
            if right[-1] > 0:
                ax['panel_F'].axhline(right[-1]-0.5, color='k', linestyle='--', lw=3)
            ax['panel_F'].axvline(10.5, color='k', linestyle='--', lw=3)
            ax['panel_F'].set_yticks([])
            ax['panel_F'].set_xlabel('time (sec)', fontsize=26)
            ax['panel_F'].set_xticks([0.5, 10.5, 20.5, 29.5])
            ax['panel_F'].set_xticklabels(np.arange(-0.5,1.1,0.5), fontsize=22)
            for k in range(right_trial_idx.shape[0]):
                ax['panel_F'].plot([right_ticks['stimOn'][k]+0.5,right_ticks['stimOn'][k]+0.5],
                                   [k-0.5,k+0.5], color='b', lw=3, 
                                   label='stim onset' if k==0 else None)
                ax['panel_F'].plot([right_ticks['feedback'][k]+0.5,right_ticks['feedback'][k]+0.5],
                                   [k-0.5,k+0.5], color='g', lw=3, 
                                   label='feedback' if k==0 else None)
            ax['panel_F'].legend(fontsize=26)

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
            cbar.ax.set_ylabel('spikes/sec', fontsize=24, labelpad=6.0)
            cbar.ax.tick_params(labelsize=20)
            
#             plt.suptitle('R2: {:.3f}, R2 on PETH: {:.3f}\nsubject={}, region={}'.format(r2, r2_psth, subject, region), fontsize=24, y=0.92)

            fg.add_labels(fig, labels)
            
            if savefig:
                figname = save_path.joinpath(f'figure9_{subject}_{region}_id={neuron}.png')
                plt.savefig(figname, bbox_inches='tight', facecolor='white')
            
            plt.close()
            

def generate_figure9_supplement1(model_config,                              
                                 preds_shape,
                                 obs,
                                 test_feature,
                                 sess_list,
                                 alpha=0.6, s=30,
                                 xlims=[4.0,45.0], ylims=[-0.075,1.0], axs=None,
                                 savefig=False):
    
#     color_names = ["windows blue",
#                    "red",
#                    "amber",
#                    "faded green",
#                    "dusty purple",
#                    "orange",
#                    "clay",
#                    "pink",
#                    "greyish",
#                    "mint",
#                    "cyan",
#                    "steel blue",
#                    "forest green",
#                    "pastel purple",
#                    "salmon",
#                    "dark brown"]
#     colors = sns.xkcd_palette(color_names)
    
    lab_number_map, institution_map, institution_colors = LAB_MAP()
    shapes = ['o', 's', '^', '+']

    mean_frs = compute_mean_frs(data_load_path.joinpath('train/shape.npy'), data_load_path.joinpath('train/output.npy'))
    
    obs_list = []
    feature_list = []
    idx = 0
    for sh in preds_shape:
        n = sh[0]*sh[1]
        obs_list.append(obs[idx:idx+n].reshape(sh[:-1]))
        feature_list.append(test_feature[idx:idx+n].reshape(sh))
        idx += n

    baseline_score = load_test_model(model_config, None, None, obs_list, preds_shape, use_psth=False)
    baseline_score_psth = load_test_model(model_config, None, None, obs_list, preds_shape, use_psth=True)

    reshaped_score = reshape_flattened(baseline_score, preds_shape, trim=3)
    reshaped_score_psth = reshape_flattened(baseline_score_psth, preds_shape, trim=3)
    reshaped_frs = reshape_flattened(mean_frs, preds_shape, trim=3)

    if axs is None:
        fig, axs = plt.subplots(1,2, sharey=True, figsize=(20,10))
        plt.subplots_adjust(wspace=0.075)
        label_size = 24
        title_size = 24
        suptitle_size = 32
        ticklabel_size = 18
        legend_size = 8.5
        s = 10
    else:
        label_size = mpl.rcParams["axes.labelsize"]
        ticklabel_size = mpl.rcParams["ytick.labelsize"]
        title_size = label_size
        suptitle_size = mpl.rcParams["axes.titlesize"]
        legend_size = 4
        s=2

    for i, sess in enumerate(sess_list):
        lab_id = np.where(feature_list[i][0,0,0,lab_offset:session_offset] == 1)[0][0]
        session_id = np.where(feature_list[i][0,0,0,session_offset:xyz_offset] == 1)[0][0]
        lab_name = sess['session']['lab']
        institution_name = institution_map[lab_name]
        institution_color = institution_colors[institution_name]
        
        axs[0].scatter(reshaped_frs[i][0], reshaped_score[i][0], s=s,
                       color=institution_color, marker=shapes[session_id], alpha=alpha,
                       label=sess['session']['subject'])
        axs[1].scatter(reshaped_frs[i][0], reshaped_score_psth[i][0], s=s,
                       color=institution_color, marker=shapes[session_id], alpha=alpha,
                       label=sess['session']['subject'])
        
        axs[0].scatter(reshaped_frs[i][1:], reshaped_score[i][1:], s=s,
                       color=institution_color, marker=shapes[session_id],
                       alpha=alpha)
        axs[1].scatter(reshaped_frs[i][1:], reshaped_score_psth[i][1:], s=s,
                       color=institution_color, marker=shapes[session_id],
                       alpha=alpha)

    axs[0].set_ylabel('R$^2$', fontsize=label_size)
    axs[0].set_xlabel('Mean firing rate (spikes/s)', fontsize=label_size)
    axs[0].set_title('Held-out test trials', fontsize=title_size)
    
    axs[1].set_xlabel('Mean firing rate (spikes/s)', fontsize=label_size)
    axs[1].set_title('PETHs of held-out test trials', fontsize=title_size)
    
    # axs[0].legend(fontsize=legend_size)
    axs[1].legend(fontsize=legend_size, bbox_to_anchor=(1, 1.05), loc='upper left', frameon=False)
    
    axs[0].set_xlim(xlims[0],xlims[1])
    axs[0].set_ylim(ylims[0],ylims[1])
    axs[1].set_xlim(xlims[0],xlims[1])
    axs[1].set_ylim(ylims[0],ylims[1])
    
    axs[0].set_yticks(np.arange(0,1.2,0.2))
    axs[0].set_yticklabels(np.arange(0,1.2,0.2), fontsize=ticklabel_size)
    axs[0].yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
    axs[1].set_yticklabels([])
    
    axs[0].set_xticks(np.arange(0,50,10).astype(int))
    axs[0].set_xticklabels(np.arange(0,50,10).astype(int), fontsize=ticklabel_size)
    
    axs[1].set_xticks(np.arange(0,50,10).astype(int))
    axs[1].set_xticklabels(np.arange(0,50,10).astype(int), fontsize=ticklabel_size)

    axs[0].text(1, 1.1, 'MTNN prediction quality vs firing rate',
                             transform=axs[0].transAxes, ha='center', fontsize=suptitle_size)
    
    if savefig:
        figname = save_path.joinpath(f'figure9_supplement1.png')
        plt.savefig(figname, bbox_inches='tight', facecolor='white')
        plt.show()
    
def generate_figure9_supplement2(model_config, 
                                 glm_score,
                                 glm_score_full_mtnn_cov,
                                 preds_shape,
                                 obs,
                                 test_feature,
                                 ax=None,
                                 savefig=False):

    obs_list = []
    feature_list = []
    idx = 0
    for sh in preds_shape:
        n = sh[0]*sh[1]
        obs_list.append(obs[idx:idx+n].reshape(sh[:-1]))
        feature_list.append(test_feature[idx:idx+n].reshape(sh))
        idx += n

    mtnn_score = load_test_model(model_config, None, None, 
                                 obs_list, preds_shape, use_psth=False, 
                                 data_dir='test', model_name_suffix=None)
    mtnn_score_glm_cov = load_test_model(model_config, leave_out_covs_for_glm, None, 
                                         obs_list, preds_shape, use_psth=False, 
                                         data_dir='test', model_name_suffix=None)

    if ax is None:
        fig = plt.figure(figsize=(10,10))
        ax = fig.fca()
        labelsize = 24
        ticklabel_size = 18
        title_size = 24
    else:
        labelsize = mpl.rcParams["axes.labelsize"]
        ticklabel_size = mpl.rcParams["ytick.labelsize"]
        title_size = mpl.rcParams["axes.titlesize"]

    
    #plt.scatter(mtnn_score[0], glm_score_full_mtnn_cov[0], color='blue', label='MTNN: Full set of covariates\nGLM: GLM covariates + behavioral covariates', alpha=0.65)
    #plt.scatter(mtnn_score_glm_cov[0], glm_score[0], color='red', label='GLM covariates', alpha=0.65)
    
    #plt.scatter(mtnn_score[1:], glm_score_full_mtnn_cov[1:], color='blue', alpha=0.65)
    #plt.scatter(mtnn_score_glm_cov[1:], glm_score[1:], color='red', alpha=0.65)
    
    ax.scatter(mtnn_score_glm_cov, glm_score_full_mtnn_cov, color='blue', alpha=0.65)
    
    ax.set_xlabel('MTNN Performance (R$^2$)', fontsize=labelsize)
    ax.set_ylabel('GLM Performance (R$^2$)', fontsize=labelsize)
    
    ax.plot([-1,1],[-1,1], color='black')
    ax.set_xlim(-0.02,0.52)
    ax.set_ylim(-0.02,0.52)
    
    ax.set_yticks(np.arange(0,0.6,0.1))#, fontsize=ticklabel_size)
    ax.set_xticks(np.arange(0,0.6,0.1))#, fontsize=ticklabel_size)
    
    ax.set_title('MTNN vs GLM predictive performance comparison\ntrained on movement/ task-related/ prior covariates',
                 fontsize=title_size)

    
    if savefig:
        figname = save_path.joinpath(f'figure9_supplement2.png')
        plt.savefig(figname,bbox_inches='tight', facecolor='white')
        plt.show()
    
def generate_figure9_supplement2_v2(model_config, 
                                     glm_score,
                                     glm_score_full_mtnn_cov,
                                     preds_shape,
                                     obs,
                                     test_feature, ax=None,
                                     savefig=False):

    obs_list = []
    feature_list = []
    idx = 0
    for sh in preds_shape:
        n = sh[0]*sh[1]
        obs_list.append(obs[idx:idx+n].reshape(sh[:-1]))
        feature_list.append(test_feature[idx:idx+n].reshape(sh))
        idx += n

    mtnn_score = load_test_model(model_config, None, None, 
                                 obs_list, preds_shape, use_psth=False, 
                                 data_dir='test', model_name_suffix=None)
    mtnn_score_glm_cov = load_test_model(model_config, leave_out_covs_for_glm, None, 
                                         obs_list, preds_shape, use_psth=False, 
                                         data_dir='test', model_name_suffix=None)

    if ax is None:
        fig = plt.figure(figsize=(10,10))
        ax = fig.gca()
        labelsize = 24
        ticklabel_size = 18
        title_size = 24
    else:
        labelsize = mpl.rcParams["axes.labelsize"]
        ticklabel_size = mpl.rcParams["ytick.labelsize"]
        title_size = mpl.rcParams["axes.titlesize"]

    
    #plt.scatter(mtnn_score[0], glm_score_full_mtnn_cov[0], color='blue', label='MTNN: Full set of covariates\nGLM: GLM covariates + behavioral covariates', alpha=0.65)
    #plt.scatter(mtnn_score_glm_cov[0], glm_score[0], color='red', label='GLM covariates', alpha=0.65)
    
    #plt.scatter(mtnn_score[1:], glm_score_full_mtnn_cov[1:], color='blue', alpha=0.65)
    #plt.scatter(mtnn_score_glm_cov[1:], glm_score[1:], color='red', alpha=0.65)
    
    plt.hist(100*(mtnn_score_glm_cov-glm_score_full_mtnn_cov)/glm_score_full_mtnn_cov, color='k')
    
    plt.xlabel('% Increase of MTNN Performance over GLM (R$^2$)', fontsize=labelsize)
    plt.ylabel('count', fontsize=labelsize)
    
    #plt.plot([-1,1],[-1,1], color='black')
    #plt.xlim(-20,20)
    #plt.ylim(-0.05,0.6)
    
    #plt.yticks(np.arange(0,0.7,0.1), fontsize=18)
    #plt.xticks(np.arange(0,0.7,0.1), fontsize=18)
    
    plt.title('MTNN vs GLM Predictive Performance Comparison\nTrained on Movement/Task-related/Prior Covariates', fontsize=title_size)
    
    plt.legend(fontsize=20)
    
    
    if savefig:
        figname = save_path.joinpath(f'figure9_supplement2_v2.png')
        plt.savefig(figname,bbox_inches='tight', facecolor='white')
    
        plt.show()
    
def get_session_order(feature_list):

    order = []
    for feature in feature_list:
        lab_id = np.where(feature[0,0,0,lab_offset:session_offset] == 1)[0][0]
        session_id = np.where(feature[0,0,0,session_offset:xyz_offset] == 1)[0][0]
        order.append((xyz_offset-session_offset)*lab_id+session_id)
        
    return np.array(order)

def generate_figure9_supplement3(model_config, 
                                 preds_shape,
                                 obs,
                                 test_feature,
                                 sess_list,
                                 preds, ax=None,
                                 savefig=False):
    
    obs_list = []
    feature_list = []
    pred_list = []
    idx = 0
    n_neurons = 0
    trial_len = obs.shape[-1]
    for sh in preds_shape:
        n = sh[0]*sh[1]
        obs_list.append(obs[idx:idx+n].reshape(sh[:-1]))
        feature_list.append(test_feature[idx:idx+n].reshape(sh))
        pred_list.append(preds[idx:idx+n].reshape(sh[:-1]))
        idx += n
        n_neurons += sh[0]

    session_order = get_session_order(feature_list)
    
    curr_idx = 0
    heatmap=np.zeros((n_neurons,4*trial_len))
    n_neurons_list = []
    for i in range(len(session_order)):
        session = np.where(session_order == i)[0][0]
        feature_i = feature_list[session]
        pred_i = pred_list[session]
        obs_i = obs_list[session]
#         print(sess_list[session])

        left_bool = feature_i[0,:,:,stimulus_offset].sum(1)!=0
        right_bool = feature_i[0,:,:,stimulus_offset+1].sum(1)!=0
        for j in range(obs_i.shape[0]):
            unit = curr_idx+j
            heatmap[unit,:trial_len] = obs_i[j,left_bool].mean(0)
            heatmap[unit,trial_len:2*trial_len] = obs_i[j,right_bool].mean(0)
            heatmap[unit,2*trial_len:3*trial_len] = pred_i[j,left_bool].mean(0)
            heatmap[unit,3*trial_len:] = pred_i[j,right_bool].mean(0)
        curr_idx += obs_i.shape[0]
        n_neurons_list.append(obs_i.shape[0])
        
    if ax is None:
        fig = plt.figure(figsize=(12,24))
        ax = fig.gca()
        labelsize = 24
        ticklabel_size = 18
        title_size = 24
        lw = 2
        text_size = 22
    else:
        labelsize = mpl.rcParams["axes.labelsize"]
        ticklabel_size = mpl.rcParams["ytick.labelsize"]
        title_size = mpl.rcParams["axes.titlesize"]
        lw = 0.5
        text_size = 7

    
    vlim = np.percentile(heatmap, [0,99.7])
    ax.imshow(heatmap, aspect='auto', cmap='Greys', interpolation='none', vmin=vlim[0], vmax=vlim[1])
    ax.axvline(x=[trial_len], linewidth=lw, linestyle='-',
                                       c='k',label='separate left and right')
    ax.axvline(x=[3*trial_len], linewidth=lw, linestyle='-',
                                       c='k',label='separate left and right')
    ax.axvline(x=[2*trial_len], linewidth=lw+0.5, linestyle='-',
                                       c='k',label='separate Y and Y_pred')
    for i in [trial_len//3,trial_len+trial_len//3,2*trial_len+trial_len//3,3*trial_len+trial_len//3]:
        ax.axvline(x=[i], linewidth=lw, linestyle='--',
                                           c='green', label='movement onset')

    session_boundary = np.cumsum(n_neurons_list)
    for i in session_boundary[:-1]:
        ax.axhline(y=[i-0.5], linewidth=lw, linestyle='--',
                                           c='red', label='session boundary')
    for idx, i in enumerate(session_boundary[:-1]):
        if (idx+1)%4 != 0:
            continue
        ax.axhline(y=[i-0.5], linewidth=lw, linestyle='-',
                                           c='blue', label='lab boundary')
        
    ax.text(trial_len-6, session_boundary[11]-5,
             'Separate left and right choice PETHs',rotation=90,color='k',fontsize=text_size)
    ax.text(3*trial_len-6, session_boundary[11]-5,
             'Separate left and right choice MTNN prediction PETHs',rotation=90,color='k',fontsize=text_size)
    
    ax.text(10-5, 0.3*n_neurons, 'Movement onset',rotation=90,color='green',fontsize=text_size-2)
    ax.text(trial_len*4-33, session_boundary[0]-1, 'Session boundary', color='red',fontsize=text_size-2)
    ax.text(trial_len*4-25, session_boundary[3]-1, 'Lab boundary', color='blue',fontsize=text_size-2)
    ax.text(10-9, 0-2, 'SWC', color='k',fontsize=text_size-2)
    ax.text(10-9, session_boundary[3]-2, 'CCU', color='k',fontsize=text_size-2)
    ax.text(10-9, session_boundary[7]-2, 'CSHL (C)', color='k',fontsize=text_size-2)
    ax.text(10-9, session_boundary[11]-2, 'UCL', color='k',fontsize=text_size-2)
    ax.text(10-9, session_boundary[15]-2, 'Berkeley', color='k',fontsize=text_size-2)
    ax.text(10-9, session_boundary[19]-2, 'NYU', color='k',fontsize=text_size-2)
    ax.text(10-9, session_boundary[23]-2, 'UCLA', color='k',fontsize=text_size-2)
    ax.text(10-9, session_boundary[27]-2, 'UW', color='k',fontsize=text_size-2)
    
    ax.set_ylabel('Neurons', fontsize=labelsize)
    ax.set_xlabel('Time (s)', fontsize=labelsize)
    ax.set_xticks([0,10,30],labels=['-0.5','0.0','1.0'], fontsize=ticklabel_size)
    #ax.set_yticks(fontsize=ticklabel_size)
    
    ax.set_title('Observed and MTNN-predicted \n PETHs on held-out trials', fontsize=title_size)
    
    if savefig:
        figname = save_path.joinpath(f'figure9_supplement3.png')
        plt.savefig(figname,bbox_inches='tight', facecolor='white')
        plt.show()
    
# def generate_figure9_supplement3(model_config, 
#                                  preds_shape,
#                                  obs,
#                                  test_feature,
#                                  sess_list,
#                                  preds,
#                                  savefig=False):
    
#     obs_list = []
#     feature_list = []
#     pred_list = []
#     idx = 0
#     n_neurons = 0
#     trial_len = obs.shape[-1]
#     for sh in preds_shape:
#         n = sh[0]*sh[1]
#         obs_list.append(obs[idx:idx+n].reshape(sh[:-1]))
#         feature_list.append(test_feature[idx:idx+n].reshape(sh))
#         pred_list.append(preds[idx:idx+n].reshape(sh[:-1]))
#         idx += n
#         n_neurons += sh[0]

#     session_order = get_session_order(feature_list)
    
#     curr_idx = 0
#     heatmap=np.zeros((n_neurons,4*trial_len))
#     n_neurons_list = []
#     for i in range(len(session_order)):
#         session = np.where(session_order == i)[0][0]
#         feature_i = feature_list[session]
#         pred_i = pred_list[session]
#         obs_i = obs_list[session]
# #         print(sess_list[session])

#         left_bool = feature_i[0,:,:,stimulus_offset].sum(1)!=0
#         right_bool = feature_i[0,:,:,stimulus_offset+1].sum(1)!=0
#         for j in range(obs_i.shape[0]):
#             unit = curr_idx+j
#             heatmap[unit,:trial_len] = obs_i[j,left_bool].mean(0)
#             heatmap[unit,trial_len:2*trial_len] = obs_i[j,right_bool].mean(0)
#             heatmap[unit,2*trial_len:3*trial_len] = pred_i[j,left_bool].mean(0)
#             heatmap[unit,3*trial_len:] = pred_i[j,right_bool].mean(0)
#         curr_idx += obs_i.shape[0]
#         n_neurons_list.append(obs_i.shape[0])
        
    
#     plt.figure(figsize=(12,24))
    
#     vlim = np.percentile(heatmap, [0,99.7])
#     plt.imshow(heatmap, aspect='auto', cmap='Greys', interpolation='none', vmin=vlim[0], vmax=vlim[1])
#     plt.axvline(x=[trial_len], linewidth=2, linestyle='-', 
#                                        c='k',label='separate left and right')
#     plt.axvline(x=[3*trial_len], linewidth=2, linestyle='-', 
#                                        c='k',label='separate left and right')
#     plt.axvline(x=[2*trial_len], linewidth=3, linestyle='-', 
#                                        c='k',label='separate Y and Y_pred')
#     for i in [trial_len//3,trial_len+trial_len//3,2*trial_len+trial_len//3,3*trial_len+trial_len//3]:
#         plt.axvline(x=[i], linewidth=2, linestyle='--', 
#                                            c='green', label='movement onset')

#     session_boundary = np.cumsum(n_neurons_list)
#     for i in session_boundary[:-1]:
#         plt.axhline(y=[i-0.5], linewidth=2, linestyle='--', 
#                                            c='red', label='session boundary')
#     for idx, i in enumerate(session_boundary[:-1]):
#         if (idx+1)%4 != 0:
#             continue
#         plt.axhline(y=[i-0.5], linewidth=3, linestyle='-', 
#                                            c='blue', label='lab boundary')
        
#     plt.text(trial_len-6, session_boundary[11]-5, 
#              'separate left and right choice PETHs',rotation=90,color='k',fontsize=22)
#     plt.text(3*trial_len-6, session_boundary[11]-5, 
#              'separate left and right choice MTNN prediction PETHs',rotation=90,color='k',fontsize=22)
    
#     plt.text(10-5, 0.3*n_neurons, 'movement onset',rotation=90,color='green',fontsize=20)
#     plt.text(trial_len*4-33, session_boundary[0]-1, 'session boundary', color='red',fontsize=20)
#     plt.text(trial_len*4-25, session_boundary[3]-1, 'lab boundary', color='blue',fontsize=20)
#     plt.text(10-9, 0-2, 'SWC', color='k',fontsize=20)
#     plt.text(10-9, session_boundary[3]-2, 'CCU', color='k',fontsize=20)
#     plt.text(10-9, session_boundary[7]-2, 'CSHL (C)', color='k',fontsize=20)
#     plt.text(10-9, session_boundary[11]-2, 'UCL', color='k',fontsize=20)
#     plt.text(10-9, session_boundary[15]-2, 'Berkeley', color='k',fontsize=20)
    
#     plt.ylabel('neurons', fontsize=18)
#     plt.xlabel('time (sec)', fontsize=18)
#     plt.xticks([0,10,30],labels=['-0.5','0.0','1.0'], fontsize=14)
#     plt.yticks(fontsize=14)
    
#     plt.title('Observed and MTNN-predicted PETHs on held-out trials', fontsize=24, y=1.03)
    
#     if savefig:
#         figname = save_path.joinpath(f'figure9_supplement3.png')
#         plt.savefig(figname,bbox_inches='tight', facecolor='white')
    
#     plt.show()

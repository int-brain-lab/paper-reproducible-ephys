import figrid as fg
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import matplotlib
import numpy as np
from copy import deepcopy

from fig_mtnn.utils import *
from fig_mtnn.mtnn import load_test_model, run_eval, initialize_mtnn, get_device

from reproducible_ephys_functions import save_data_path, figure_style, save_figure_path, labs

import torch

data_load_path = save_data_path(figure='fig_mtnn').joinpath('mtnn_data')
save_path = save_figure_path(figure='fig_mtnn')


regions = list(get_acronym_dict().keys())
acronym_dict_reverse = get_acronym_dict_reverse()
region_colors = get_region_colors()

dynamic_covs = ['paw speed', 'nose speed', 'pupil diameter', 
                'motion energy', 'stimuli', 'go cue', 'first movement',
                'choice', 'reward', 'wheel velocity', 'lick', 'noise']

static_covs = ['mouse prior', 'last mouse prior', 'decision strategy (GLM-HMM)']

def make_fig_ax():
    
    xsplit = ([0.03,0.4], [0.46,0.89], [0.90,1.0])
    ysplit = ([0.03,1],)
    
    fig = plt.figure(figsize=(30,8))
    ax = {'panel_A': fg.place_axes_on_grid(fig, xspan=xsplit[0], yspan=ysplit[0]),
          'panel_B': fg.place_axes_on_grid(fig, xspan=xsplit[1], yspan=ysplit[0]),
          'panel_C': fg.place_axes_on_grid(fig, xspan=xsplit[2], yspan=ysplit[0])}
    
    # Add subplot labels
    labels = [{'label_text':'a', 'xpos':0, 'ypos':0, 'fontsize':30, 'weight': 'bold',
               'ha': 'right', 'va': 'bottom'},
              {'label_text':'b', 'xpos':0.44, 'ypos':0, 'fontsize':30, 'weight': 'bold',
               'ha': 'right', 'va': 'bottom'}]
    
#     xsplit = ([0.03,0.47], [0.52,1])
#     ysplit = ([0.03,0.47], [0.52,1])
    
#     fig = plt.figure(figsize=(30,20))
#     ax = {'panel_A': fg.place_axes_on_grid(fig, xspan=xsplit[0], yspan=ysplit[0]),
#           'panel_B': fg.place_axes_on_grid(fig, xspan=xsplit[1], yspan=ysplit[0]),
#           'panel_C': fg.place_axes_on_grid(fig, xspan=xsplit[0], yspan=ysplit[1]),
#           'panel_D': fg.place_axes_on_grid(fig, xspan=xsplit[1], yspan=ysplit[1])}
    
#     # Add subplot labels
#     labels = [{'label_text':'a', 'xpos':0, 'ypos':0, 'fontsize':30, 'weight': 'bold',
#                'ha': 'right', 'va': 'bottom'},
#               {'label_text':'b', 'xpos':0.5, 'ypos':0, 'fontsize':30, 'weight': 'bold',
#                'ha': 'right', 'va': 'bottom'},
#               {'label_text':'c', 'xpos':0, 'ypos':0.5, 'fontsize':30, 'weight': 'bold',
#                'ha': 'right', 'va': 'bottom'},
#               {'label_text':'d', 'xpos':0.5, 'ypos':0.5, 'fontsize':30, 'weight': 'bold',
#                'ha': 'right', 'va': 'bottom'}]
    
    return fig, ax, labels



def get_region_inds(feature_list):
    
    start, end = cov_idx_dict['brain region']
    
    region_inds = []
    for feature in feature_list:
        n_neurons = feature.shape[0]
        for neu_id in range(n_neurons):
            region_idx = np.nonzero(feature[neu_id,0,0,start:end])[0][0]
            region_inds.append(acronym_dict_reverse[region_idx])
    return np.asarray(region_inds)

def compute_scores_for_figure_10(model_config,
                                 leave_one_out_covs,
                                 single_covs,
                                 leave_group_out,
                                 use_psth=False):
    
    load_path = save_data_path(figure='fig_mtnn').joinpath('mtnn_data')
    
    preds_shape = np.load(load_path.joinpath('test/shape.npy'))
    obs = np.load(load_path.joinpath('test/output.npy'))
    test_feature = np.load(load_path.joinpath('test/feature.npy'))
    mean_frs = compute_mean_frs(data_load_path.joinpath('train/shape.npy'), data_load_path.joinpath('train/output.npy'))
    
    obs_list = []
    feature_list = []
    idx = 0
    for sh in preds_shape:
        n = sh[0]*sh[1]
        obs_list.append(obs[idx:idx+n].reshape(sh[:-1]))
        feature_list.append(test_feature[idx:idx+n].reshape(sh))
        idx += n
        
    region_inds = get_region_inds(feature_list)   
    baseline_score = load_test_model(model_config, None, 
                                     None, obs_list, preds_shape, use_psth=use_psth)
    baseline_score2 = load_test_model(model_config, ['session'], 
                                      None, obs_list, preds_shape, use_psth=use_psth)

    frs = {}
    scores = {}
    scores['leave_one_out'] = {}
    scores['single_covariate'] = {}
    scores['leave_group_out'] = {}
    for i, cov in enumerate(leave_one_out_covs):
        cov = [cov]
        scores['leave_one_out'][tuple(cov)]={}
        score = load_test_model(model_config, cov, None, obs_list, preds_shape, use_psth=use_psth)
        if cov[0] == 'lab':
            scores['leave_one_out'][tuple(cov)]['all'] = baseline_score2-score
        else:
            scores['leave_one_out'][tuple(cov)]['all'] = baseline_score-score
        for region in regions:
            region_score = np.where(region_inds==region)
            
            if cov[0] == 'lab':
                score_diff = baseline_score2[region_score]-score[region_score]
            else:
                score_diff = baseline_score[region_score]-score[region_score]
            scores['leave_one_out'][tuple(cov)][region] = score_diff
            
    for i, cov in enumerate(leave_group_out):
        scores['leave_group_out'][tuple(cov)]={}
        score = load_test_model(model_config, cov, None, obs_list, preds_shape, use_psth=use_psth)
        scores['leave_group_out'][tuple(cov)]['all'] = baseline_score-score
        for region in regions:
            region_score = np.where(region_inds==region)
            score_diff = baseline_score[region_score]-score[region_score]
            scores['leave_group_out'][tuple(cov)][region] = score_diff
                
    model_config_dynamic = deepcopy(model_config)
    model_config_dynamic['static_bias'] = False
    model_config_static = deepcopy(model_config)
    model_config_static['dynamic_bias'] = False
    for i, cov in enumerate(single_covs):
        scores['single_covariate'][cov]={}
        if cov in dynamic_covs:
            score = load_test_model(model_config_dynamic, None, cov, obs_list, 
                                    preds_shape, use_psth=use_psth)
            scores['single_covariate'][cov]['all'] = score
        else:
            score = load_test_model(model_config_static, None, cov, obs_list,
                                    preds_shape, use_psth=use_psth)
            scores['single_covariate'][cov]['all'] = score
        for region in regions:
            region_score = np.where(region_inds==region)
            score_diff = scores['single_covariate'][cov]['all'][region_score]
            scores['single_covariate'][cov][region] = score_diff
    
    for region in regions:
        region_score = np.where(region_inds==region)
        frs[region] = mean_frs[region_score]
    
    return scores, frs

def generate_figure_10(model_config,
                       leave_one_out_covs,
                       single_covs,
                       leave_group_out,
                       savefig=False):
    
    scores, frs = compute_scores_for_figure_10(model_config,
                                              leave_one_out_covs,
                                              single_covs,
                                              leave_group_out)

    figure_style()
    
    fig, ax, labels = make_fig_ax()
    
#     ax['panel_A'].set_xticks([])
#     ax['panel_B'].set_xticks([])
    ax['panel_A'].set_title('Single-covariate analysis', fontsize=36)
    ax['panel_B'].set_title('Leave-one-out analysis', fontsize=36)
    ax['panel_C'].set_title('Leave-group-out', fontsize=36)
    ax['panel_A'].set_ylabel('R2', fontsize=36)
#     ax['panel_C'].set_ylabel('R2')
    ax['panel_B'].set_ylabel(r'$\Delta$'+'R2', fontsize=36)
#     ax['panel_D'].set_ylabel(r'$\Delta$'+'R2')
    
    # Leave-one-out
    all_scores_mean_list = []
    leave_one_out_covs_renamed = []
    for i in range(len(leave_one_out_covs)):
        cov = leave_one_out_covs[i]
        if cov == 'decision strategy (GLM-HMM)':
            leave_one_out_covs_renamed.append('decision strategy')
        else:
            leave_one_out_covs_renamed.append(cov)
        
        cov = [cov]
        if cov[0] == 'noise':
            all_scores_mean_list.append(-np.Inf)
        else:
            all_scores_mean = scores['leave_one_out'][tuple(cov)]['all'].mean()
            all_scores_mean_list.append(all_scores_mean)
    leave_one_out_ordering = np.argsort(all_scores_mean_list)[::-1]
    
#     leave_one_out_covs = list(scores['leave_one_out'].keys())
#     print(leave_one_out_covs)
    ax['panel_B'].set_xticks(np.arange(len(leave_one_out_covs)))
    ax['panel_B'].set_xticklabels(np.array(leave_one_out_covs_renamed)[leave_one_out_ordering], 
                                  rotation=90, fontsize=32)
    ax['panel_B'].set_xlim(-1,len(leave_one_out_covs))
#     ax['panel_D'].set_xlim(-1,len(leave_one_out_covs))
    ax['panel_B'].axhline(0, color='k', linestyle='--', lw=3)
#     ax['panel_D'].axhline(0, color='k', linestyle='--')

    for i in range(len(leave_one_out_covs)):
        cov = leave_one_out_covs[leave_one_out_ordering[i]]
        cov = [cov]
        all_scores_mean = all_scores_mean_list[leave_one_out_ordering[i]]
#         ax['panel_B'].plot([i-0.3,i+0.3], [all_scores_mean,all_scores_mean], 
#                            color='k',linestyle='--')
        for j, region in enumerate(regions):
            scr = scores['leave_one_out'][tuple(cov)][region]
            ax['panel_B'].scatter(np.ones_like(scr)*i+0.1*(j-2), scr, 
                                  color=region_colors[region], label=region if i==0 else None, s=24, alpha=0.6)
#             ax['panel_D'].scatter(np.ones_like(scr)*i+0.1*(j-2), scr, 
#                                   c=frs[region], cmap=plt.get_cmap('Reds'), s=20, alpha=0.7)
    #ax['panel_B'].legend(fontsize=24)
#     ax['panel_B'].set_ylim(-0.1,0.2)
    ax['panel_B'].set_ylim(-0.2,0.5)
    ax['panel_B'].set_yticks(np.arange(-0.1,0.6,0.1))
    ax['panel_B'].set_yticklabels(np.arange(-0.1,0.6,0.1), fontsize=18)
    ax['panel_B'].yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
#     ax['panel_D'].set_ylim(-0.1,0.2)
    
    # Leave-group-out
    all_scores_mean_list = []
    for i in range(len(leave_group_out)):
        cov = leave_group_out[i]
        all_scores_mean = scores['leave_group_out'][tuple(cov)]['all'].mean()
        all_scores_mean_list.append(all_scores_mean)
    leave_group_out_ordering = np.argsort(all_scores_mean_list)[::-1]

    leave_group_out_covs = list(scores['leave_group_out'].keys())
    leave_group_out_ticklabels = []
    for group in leave_group_out_covs:
        for key in grouped_cov_idx_dict.keys():
            if set(grouped_cov_idx_dict[key]) == set(group):
                if key == 'task':
                    leave_group_out_ticklabels.append('task-related')
                elif key == 'ephys':
                    leave_group_out_ticklabels.append('electrophysiological')
                else:
                    leave_group_out_ticklabels.append(key)
    
    ax['panel_C'].set_xticks(np.arange(len(leave_group_out_covs)))
    ax['panel_C'].set_xticklabels(np.array(leave_group_out_ticklabels)[leave_group_out_ordering], 
                                  rotation=90, fontsize=32)
    ax['panel_C'].set_xlim(-1,len(leave_group_out_covs))
    ax['panel_C'].axhline(0, color='k', linestyle='--', lw=3)
    
    for i in range(len(leave_group_out)):
        cov = leave_group_out[leave_group_out_ordering[i]]
        all_scores_mean = all_scores_mean_list[leave_group_out_ordering[i]]
        for j, region in enumerate(regions):
            scr = scores['leave_group_out'][tuple(cov)][region]
            ax['panel_C'].scatter(np.ones_like(scr)*i+0.1*(j-2), scr, 
                                  color=region_colors[region], label=region if i==0 else None, s=24, alpha=0.6)
    #ax['panel_C'].legend(fontsize=24)
#     ax['panel_C'].set_ylim(-0.1,0.2)
    ax['panel_C'].set_ylim(-0.2,0.5)
    ax['panel_C'].set_yticks([])
    
    # Single-covariate
    all_scores_mean_list = []
    single_covs_renamed = []
    for i in range(len(single_covs)):
        cov = single_covs[i]
        if cov == 'decision strategy (GLM-HMM)':
            single_covs_renamed.append('decision strategy')
        else:
            single_covs_renamed.append(cov)
            
        if cov == 'noise':
            all_scores_mean_list.append(-np.Inf)
        else:
            all_scores_mean = scores['single_covariate'][cov]['all'].mean()
            all_scores_mean_list.append(all_scores_mean)
    single_cov_ordering = np.argsort(all_scores_mean_list)[::-1]
    
#     single_covs = list(scores['single_covariate'].keys())
    ax['panel_A'].set_xticks(np.arange(len(single_covs)))
    ax['panel_A'].set_xticklabels(np.array(single_covs_renamed)[single_cov_ordering], 
                                  rotation=90, fontsize=32)
    ax['panel_A'].set_xlim(-1,len(single_covs))
#     ax['panel_C'].set_xlim(-1,len(single_covs))
    ax['panel_A'].axhline(0, color='k', linestyle='--', lw=3)
#     ax['panel_C'].axhline(0, color='k', linestyle='--')

    for i in range(len(single_covs)):
        cov = single_covs[single_cov_ordering[i]]
        all_scores_mean = all_scores_mean_list[single_cov_ordering[i]]
        print(single_cov_ordering[i], all_scores_mean)
#         ax['panel_A'].plot([i-0.3,i+0.3], [all_scores_mean,all_scores_mean], 
#                            color='k',linestyle='--')
        for j, region in enumerate(regions):
            scr = scores['single_covariate'][cov][region]
            color = region_colors[region]
            label = region if region != 'PPC' else 'VISa/am'
            ax['panel_A'].scatter(np.ones_like(scr)*i+0.1*(j-2), scr, 
                                  color=color, label=label if i==0 else None, s=24, alpha=0.6)
#             ax['panel_C'].scatter(np.ones_like(scr)*i+0.1*(j-2), scr, 
#                                   c=frs[region], cmap=plt.get_cmap('Reds'), s=20, alpha=0.7)
    ax['panel_A'].legend(fontsize=24)
    ax['panel_A'].set_ylim(-0.2,0.5)
    ax['panel_A'].set_yticks(np.arange(-0.1,0.6,0.1))
    ax['panel_A'].set_yticklabels(np.arange(-0.1,0.6,0.1), fontsize=18)
    ax['panel_A'].yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
#     ax['panel_C'].set_ylim(-0.1,0.5)

    
    fg.add_labels(fig, labels)
    if savefig:
        figname = save_path.joinpath(f'figure10.png')
        plt.savefig(figname, bbox_inches='tight', facecolor='white')
    
    plt.show()
    
def make_fig_ax_supplement1():
    
    xsplit = ([0.375,0.625], [0,1])
    ysplit = ([0,0.3],[0.4,1.0])
    
    fig = plt.figure(figsize=(30,24))
    ax = {'panel_A': fg.place_axes_on_grid(fig, xspan=xsplit[0], yspan=ysplit[0]),
          'panel_B': fg.place_axes_on_grid(fig, xspan=xsplit[1], yspan=ysplit[1],
                                           dim=[2, 4], sharey=True,
                                           wspace=0.08, hspace=0.08)}
    
    # Add subplot labels
    labels = [{'label_text':'a', 'xpos':0, 'ypos':0, 'fontsize':30, 'weight': 'bold',
               'ha': 'right', 'va': 'bottom'},
              {'label_text':'b', 'xpos':0, 'ypos':0.37, 'fontsize':30, 'weight': 'bold',
               'ha': 'right', 'va': 'bottom'}]
    return fig, ax, labels
    
def generate_figure_10_supplement1(model_config,
                                   glm_scores,
                                   glm_leave_one_out,
                                   savefig=False,
                                   down_lim=-0.05,
                                   up_lim=0.75):
    
    load_path = save_data_path(figure='fig_mtnn').joinpath('simulated_data')
    
    data_dir = 'test'
    preds_shape = np.load(load_path.joinpath(f'{data_dir}/shape.npy'))
    obs = np.load(load_path.joinpath(f'{data_dir}/output.npy'))
    test_feature = np.load(load_path.joinpath(f'{data_dir}/feature.npy'))

    obs_list = []
    feature_list = []
    idx = 0
    for sh in preds_shape:
        n = sh[0]*sh[1]
        obs_list.append(obs[idx:idx+n].reshape(sh[:-1]))
        feature_list.append(test_feature[idx:idx+n].reshape(sh))
        idx += n

    best_score = load_test_model(model_config, None, None, 
                                 obs_list, preds_shape, use_psth=False, 
                                 data_dir=data_dir, model_name_suffix=None, simulated=True)
    leftstim_score = load_test_model(model_config, ['left stimuli'], None, 
                                 obs_list, preds_shape, use_psth=False, 
                                 data_dir=data_dir, model_name_suffix=None, simulated=True)
    rightstim_score = load_test_model(model_config, ['right stimuli'], None, 
                                 obs_list, preds_shape, use_psth=False, 
                                 data_dir=data_dir, model_name_suffix=None, simulated=True)
    incorrect_score = load_test_model(model_config, ['incorrect'], None, 
                                 obs_list, preds_shape, use_psth=False, 
                                 data_dir=data_dir, model_name_suffix=None, simulated=True)
    correct_score = load_test_model(model_config, ['correct'], None, 
                                 obs_list, preds_shape, use_psth=False, 
                                 data_dir=data_dir, model_name_suffix=None, simulated=True)
    fmv_score = load_test_model(model_config, ['first movement'], None, 
                                 obs_list, preds_shape, use_psth=False, 
                                 data_dir=data_dir, model_name_suffix=None, simulated=True)
    prior_score = load_test_model(model_config, ['mouse prior'], None, 
                                 obs_list, preds_shape, use_psth=False, 
                                 data_dir=data_dir, model_name_suffix=None, simulated=True)
    last_prior_score = load_test_model(model_config, ['last mouse prior'], None, 
                                 obs_list, preds_shape, use_psth=False, 
                                 data_dir=data_dir, model_name_suffix=None, simulated=True)
    wheel_score = load_test_model(model_config, ['wheel velocity'], None, 
                                 obs_list, preds_shape, use_psth=False, 
                                 data_dir=data_dir, model_name_suffix=None, simulated=True)

    figure_style()
    
    fig, ax, labels = make_fig_ax_supplement1()
    ax['panel_B'][0][0].set_xticks([])
    ax['panel_B'][0][1].set_xticks([])
    ax['panel_B'][0][2].set_xticks([])
    ax['panel_B'][0][3].set_xticks([])
    
    ax['panel_A'].scatter(best_score, glm_scores, color='k', alpha=0.6)
    ax['panel_A'].set_xlim(down_lim, up_lim)
    ax['panel_A'].set_ylim(down_lim, up_lim)
    ax['panel_A'].plot([-1,1],[-1,1],color='k')
    ax['panel_A'].set_ylabel('GLM predictive performance (R2)', fontsize=20)
    ax['panel_A'].set_xlabel('MTNN predictive performance (R2)', fontsize=20)
    ax['panel_A'].set_yticks(np.arange(0,0.8,0.1))
    ax['panel_A'].set_yticklabels(np.arange(0,0.8,0.1), fontsize=14)
    ax['panel_A'].yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
    ax['panel_A'].set_xticks(np.arange(0,0.8,0.1))
    ax['panel_A'].set_xticklabels(np.arange(0,0.8,0.1), fontsize=14)
    ax['panel_A'].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
    ax['panel_A'].set_title('GLM vs MTNN\npredictive performance on simulated data', fontsize=24)
    
    ax['panel_B'][0][0].set_title('left stimulus', fontsize=16)
    ax['panel_B'][0][1].set_title('right stimulus', fontsize=16)
    ax['panel_B'][0][2].set_title('incorrect', fontsize=16)
    ax['panel_B'][0][3].set_title('correct', fontsize=16)
    ax['panel_B'][1][0].set_title('first movement onset', fontsize=16)
    ax['panel_B'][1][1].set_title('mouse prior', fontsize=16)
    ax['panel_B'][1][2].set_title('previous mouse prior', fontsize=16)
    ax['panel_B'][1][3].set_title('wheel velocity', fontsize=16)
    
    ax['panel_B'][0][0].set_ylabel('GLM Effect Size '+r'($\Delta$'+'R2)', fontsize=16)
    ax['panel_B'][1][0].set_ylabel('GLM Effect Size '+r'($\Delta$'+'R2)', fontsize=16)
    
    ax['panel_B'][1][0].set_xlabel('MTNN Effect Size '+r'($\Delta$'+'R2)', fontsize=16)
    ax['panel_B'][1][1].set_xlabel('MTNN Effect Size '+r'($\Delta$'+'R2)', fontsize=16)
    ax['panel_B'][1][2].set_xlabel('MTNN Effect Size '+r'($\Delta$'+'R2)', fontsize=16)
    ax['panel_B'][1][3].set_xlabel('MTNN Effect Size '+r'($\Delta$'+'R2)', fontsize=16)
    
    ax['panel_B'][0][0].scatter(best_score-leftstim_score, glm_leave_one_out[:,0], color='k', alpha=0.6)
    ax['panel_B'][0][0].set_xlim(down_lim,up_lim)
    ax['panel_B'][0][0].set_ylim(down_lim,up_lim)
    ax['panel_B'][0][0].plot([-1,1],[-1,1],color='k')
    ax['panel_B'][0][0].set_yticks(np.arange(0,0.8,0.1))
    ax['panel_B'][0][0].set_yticklabels(np.arange(0,0.8,0.1), fontsize=14)
    ax['panel_B'][0][0].yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
    
    ax['panel_B'][0][1].scatter(best_score-rightstim_score, glm_leave_one_out[:,1], color='k', alpha=0.6)
    ax['panel_B'][0][1].set_xlim(down_lim,up_lim)
    ax['panel_B'][0][1].set_ylim(down_lim,up_lim)
    ax['panel_B'][0][1].plot([-1,1],[-1,1],color='k')
    
    ax['panel_B'][0][2].scatter(best_score-incorrect_score, glm_leave_one_out[:,3], color='k', alpha=0.6)
    ax['panel_B'][0][2].set_xlim(down_lim,up_lim)
    ax['panel_B'][0][2].set_ylim(down_lim,up_lim)
    ax['panel_B'][0][2].plot([-1,1],[-1,1],color='k')
    
    ax['panel_B'][0][3].scatter(best_score-correct_score, glm_leave_one_out[:,2], color='k', alpha=0.6)
    ax['panel_B'][0][3].set_xlim(down_lim,up_lim)
    ax['panel_B'][0][3].set_ylim(down_lim,up_lim)
    ax['panel_B'][0][3].plot([-1,1],[-1,1],color='k')
    
    ax['panel_B'][1][0].scatter(best_score-fmv_score, glm_leave_one_out[:,4], color='k', alpha=0.6)
    ax['panel_B'][1][0].set_xlim(down_lim,up_lim)
    ax['panel_B'][1][0].set_ylim(down_lim,up_lim)
    ax['panel_B'][1][0].plot([-1,1],[-1,1],color='k')
    ax['panel_B'][1][0].set_yticks(np.arange(0,0.8,0.1))
    ax['panel_B'][1][0].set_yticklabels(np.arange(0,0.8,0.1), fontsize=14)
    ax['panel_B'][1][0].yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
    ax['panel_B'][1][0].set_xticks(np.arange(0,0.8,0.1))
    ax['panel_B'][1][0].set_xticklabels(np.arange(0,0.8,0.1), fontsize=14)
    ax['panel_B'][1][0].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
    
    ax['panel_B'][1][1].scatter(best_score-prior_score, glm_leave_one_out[:,5], color='k', alpha=0.6)
    ax['panel_B'][1][1].set_xlim(down_lim,up_lim)
    ax['panel_B'][1][1].set_ylim(down_lim,up_lim)
    ax['panel_B'][1][1].plot([-1,1],[-1,1],color='k')
    ax['panel_B'][1][1].set_xticks(np.arange(0,0.8,0.1))
    ax['panel_B'][1][1].set_xticklabels(np.arange(0,0.8,0.1), fontsize=14)
    ax['panel_B'][1][1].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
    
    ax['panel_B'][1][2].scatter(best_score-last_prior_score, glm_leave_one_out[:,6], color='k', alpha=0.6)
    ax['panel_B'][1][2].set_xlim(down_lim,up_lim)
    ax['panel_B'][1][2].set_ylim(down_lim,up_lim)
    ax['panel_B'][1][2].plot([-1,1],[-1,1],color='k')
    ax['panel_B'][1][2].set_xticks(np.arange(0,0.8,0.1))
    ax['panel_B'][1][2].set_xticklabels(np.arange(0,0.8,0.1), fontsize=14)
    ax['panel_B'][1][2].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
    
    ax['panel_B'][1][3].scatter(best_score-wheel_score, glm_leave_one_out[:,7], color='k', alpha=0.6)
    ax['panel_B'][1][3].set_xlim(down_lim,up_lim)
    ax['panel_B'][1][3].set_ylim(down_lim,up_lim)
    ax['panel_B'][1][3].plot([-1,1],[-1,1],color='k')
    ax['panel_B'][1][3].set_xticks(np.arange(0,0.8,0.1))
    ax['panel_B'][1][3].set_xticklabels(np.arange(0,0.8,0.1), fontsize=14)
    ax['panel_B'][1][3].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
    
    plt.suptitle('GLM vs MTNN Effect Sizes on Simulated Data', y=0.61, fontsize=24)
    
    fg.add_labels(fig, labels)
    if savefig:
        figname = save_path.joinpath(f'figure10_supplement1.png')
        plt.savefig(figname, bbox_inches='tight', facecolor='white')
    
    plt.show()

def generate_figure_10_supplement2(model_config,
                                   single_covs,
                                   savefig=False):
    
#     color_names = ["windows blue",
#                    "amber",
#                    "faded green",
#                    "red",
#                    "dusty purple",
#                    "black",
#                    "magenta",
#                    "cyan"]

#     colors = sns.xkcd_palette(color_names)

    lab_number_map, institution_map, institution_colors = labs()

    shapes = ['o', '+', '^', 's']
    
    scores, frs = compute_scores_for_figure_10(model_config,
                                              [],
                                              single_covs,
                                              [])

    preds_shape = np.load(data_load_path.joinpath('test/shape.npy'))
    sess_list = np.load(data_load_path.joinpath('session_info.npy'), allow_pickle=True).tolist()
    
    all_scores_mean_list = []
    single_covs_renamed = []
    for i in range(len(single_covs)):
        cov = single_covs[i]
        if cov == 'decision strategy (GLM-HMM)':
            single_covs_renamed.append('decision strategy')
        else:
            single_covs_renamed.append(cov)
            
        if cov == 'noise':
            all_scores_mean_list.append(-np.Inf)
        else:
            all_scores_mean = scores['single_covariate'][cov]['all'].mean()
            all_scores_mean_list.append(all_scores_mean)
    single_cov_ordering = np.argsort(all_scores_mean_list)[::-1]
    
    ncovs = len(single_covs)
    fig, axs = plt.subplots(ncovs, ncovs, figsize=(24,24), sharey=True, sharex=True)
    plt.subplots_adjust(wspace=0.03, hspace=0.03)
    
    for i in range(ncovs):
        covi = single_covs[single_cov_ordering[i]]
        for j in range(ncovs):
            covj = single_covs[single_cov_ordering[j]]        
                
            if j == 0:
                axs[i,j].set_ylabel(covi, rotation=45, fontsize=20)
                axs[i,j].yaxis.set_label_coords(-0.9, 0.5)
                axs[i,j].set_yticks(np.arange(0,0.6,0.2))
                axs[i,j].set_yticklabels(np.arange(0,0.6,0.2), fontsize=14)
                axs[i,j].yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
            if i == ncovs-1:
                axs[i,j].set_xlabel(covj, rotation=45, fontsize=20)
                axs[i,j].xaxis.set_label_coords(0.5, -0.2)
                axs[i,j].set_xticks(np.arange(0,0.6,0.2))
                axs[i,j].set_xticklabels(np.arange(0,0.6,0.2), fontsize=14)
                axs[i,j].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
            
            if i==0 and j==0:
                for n in range(8):
                    for m in range(4):
                        subject = sess_list[4*n+m]['session']['subject']#.tolist()['session']['subject']
                        lab_name = sess_list[4*n+m]['session']['lab']
                        institution_name = institution_map[lab_name]
                        institution_color = institution_colors[institution_name]
                        axs[i,j].scatter(-2, -2, color=institution_color, marker=shapes[m], 
                                         alpha=0.8, s=70, label=subject)
                axs[i,j].legend(bbox_to_anchor=(13.0,-5.5), fontsize=18)
                continue
            elif i == j:
                continue
                
            scorei_list = reshape_flattened(scores['single_covariate'][covi]['all'], preds_shape, trim=3)
            scorej_list = reshape_flattened(scores['single_covariate'][covj]['all'], preds_shape, trim=3)
            
            for n in range(8):
                for m in range(4):
                    lab_name = sess_list[4*n+m]['session']['lab']
                    institution_name = institution_map[lab_name]
                    institution_color = institution_colors[institution_name]
                    axs[i,j].scatter(scorej_list[4*n+m],
                                     scorei_list[4*n+m], color=institution_color, marker=shapes[m], s=30, alpha=0.5)
            axs[i,j].plot([-1,1],[-1,1],color='black')
            axs[i,j].set_ylim(-0.1,0.55)
            axs[i,j].set_xlim(-0.1,0.55)

    plt.suptitle('Pairwise scatterplots of MTNN single-covariate effect sizes', y=0.92, fontsize=40)
    
    if savefig:
        figname = save_path.joinpath(f'figure10_supplement2.pdf')
        plt.savefig(figname, bbox_inches='tight', facecolor='white', dpi=600)
    
    plt.show()

    
def generate_figure_10_supplement3(model_config, savefig=False):
    
    lab_number_map, institution_map, institution_colors = labs()

    data_path = save_data_path(figure='fig_mtnn')
    data_load_path = data_path.joinpath('mtnn_data')
    test_feature = np.load(data_load_path.joinpath('test/feature.npy'))
    test_trials = np.load(data_load_path.joinpath('test/trials.npy'), allow_pickle=True)
    
    neuron_order = test_feature[:, 0, 0]
    feature = test_feature[:, :, 1:]

    neurons = np.unique(neuron_order)
    n_neurons = neurons.shape[0]
    print('number of neurons: {}'.format(n_neurons))

    INPUT_SIZE_DYNAMIC = feature.shape[-1] - static_idx.shape[0]
    INPUT_SIZE_STATIC = static_idx.shape[0]
    print(INPUT_SIZE_STATIC, INPUT_SIZE_DYNAMIC)

    HIDDEN_SIZE_STATIC = 128
    HIDDEN_SIZE_DYNAMIC = 128
    n_layers = 4
    dropout = 0.15

    remove_cov = None
    only_keep_cov = None

    model = initialize_mtnn(n_neurons=n_neurons,
                            input_size_static=INPUT_SIZE_STATIC,
                            input_size_dynamic=INPUT_SIZE_DYNAMIC,
                            static_bias=True, dynamic_bias=True,
                            hidden_dim_static=HIDDEN_SIZE_STATIC,
                            hidden_dim_dynamic=HIDDEN_SIZE_DYNAMIC, n_layers=n_layers,
                            dropout=dropout)

    model_load_path = data_path.joinpath(f'trained_models/state_dict_rem={remove_cov}_keep={only_keep_cov}.pt')
    model.load_state_dict(torch.load(model_load_path))

    preds, loss = run_eval(model, data_load_path.joinpath('test/feature.npy'),
                           data_load_path.joinpath('test/output.npy'),
                           remove_cov=remove_cov, only_keep_cov=only_keep_cov)

    preds_shape = np.load(data_load_path.joinpath('test/shape.npy'))
    obs = np.load(data_load_path.joinpath('test/output.npy'))
    test_feature = np.load(data_load_path.joinpath('test/feature.npy'))
    neu_list = np.load(data_load_path.joinpath('clusters.npy'), allow_pickle=True)
    sess_list = np.load(data_load_path.joinpath('session_info.npy'), allow_pickle=True).tolist()
    trial_list = np.load(data_load_path.joinpath('test/trials.npy'), allow_pickle=True)

    pred_list = []
    obs_list = []
    feature_list = []
    idx = 0
    for sh in preds_shape:
        n = sh[0] * sh[1]
        pred_list.append(preds[idx:idx + n].reshape(sh[:-1]))
        obs_list.append(obs[idx:idx + n].reshape(sh[:-1]))
        feature_list.append(test_feature[idx:idx + n].reshape(sh))
        idx += n
    
    # panel 1: region PO/observed
    plt.figure(figsize=(6,4))
    n_sessions = len(obs_list)
    plotted_institutions = []
    for i in range(n_sessions):
        feat = feature_list[i]
        obs = obs_list[i]
        sess = sess_list[i]
        PO_idx = feat[:, 0, 0, acronym_offset+4] == 1
        lab_name = sess['session']['lab']
        session_name = sess['session']['subject']
        institution_name = institution_map[lab_name]
        institution_color = institution_colors[institution_name]

        if PO_idx.sum() == 0:
            continue

        mean_fr = np.mean(obs[PO_idx], axis=(0,1))
        baselined_mean_fr = mean_fr-mean_fr[0]
        if institution_name in plotted_institutions:
            plt.plot(baselined_mean_fr, color=institution_color, linewidth=2)
        else:
            plt.plot(baselined_mean_fr, label=institution_name, color=institution_color, linewidth=2)
        plotted_institutions.append(institution_name)
    plt.ylim(-6, 12.5)
    plt.ylabel('Baselined firing rate (sp/s)')
    plt.xlabel('Time from movement onset (s)')
    plt.axvline(10, color='k', linestyle=':')
    plt.xticks(np.arange(0, obs_list[0].shape[-1]+1, 10), labels=[-0.5, 0.0, 0.5, 1.0])
    plt.legend(ncols=1)
    plt.title('Observed PETH of Held-out Test Trials')
    if savefig:
        figname = save_path.joinpath(f'original_peth.png')
        plt.savefig(figname, bbox_inches='tight', facecolor='white', dpi=600)
    plt.show()
    
    
    # panel 2: region PO/predicted
    plt.figure(figsize=(6,4))
    n_sessions = len(obs_list)
    plotted_institutions = []
    for i in range(n_sessions):
        feat = feature_list[i]
        pred = pred_list[i]
        sess = sess_list[i]
        PO_idx = feat[:, 0, 0, acronym_offset+4] == 1
        lab_name = sess['session']['lab']
        session_name = sess['session']['subject']
        institution_name = institution_map[lab_name]
        institution_color = institution_colors[institution_name]

        if PO_idx.sum() == 0:
            continue

        mean_fr = np.mean(pred[PO_idx], axis=(0,1))
        baselined_mean_fr = mean_fr-mean_fr[0]
        if institution_name in plotted_institutions:
            plt.plot(baselined_mean_fr, color=institution_color, linewidth=2)
        else:
            plt.plot(baselined_mean_fr, label=institution_name, color=institution_color, linewidth=2)
        plotted_institutions.append(institution_name)
    plt.ylim(-6, 12.5)
    plt.ylabel('Baselined firing rate (sp/s)')
    plt.xlabel('Time from movement onset (s)')
    plt.axvline(10, color='k', linestyle=':')
    plt.xticks(np.arange(0, obs_list[0].shape[-1]+1, 10), labels=[-0.5, 0.0, 0.5, 1.0])
    plt.legend(ncols=1)
    plt.title('Predicted PETH of Held-out Test Trials')
    if savefig:
        figname = save_path.joinpath(f'pred_peth.png')
        plt.savefig(figname, bbox_inches='tight', facecolor='white', dpi=600)
    plt.show()
    
    
    
    fake_feature = deepcopy(test_feature)
    fake_feature[:, :, lab_offset:session_offset] = 0
    fake_feature[:, :, lab_offset+1] = 1 # CCU

    fake_preds, loss = run_eval(model, None,
                                data_load_path.joinpath('test/output.npy'),
                                test_feature=fake_feature,
                                remove_cov=remove_cov, only_keep_cov=only_keep_cov)

    fake_pred_list = []
    idx = 0
    for sh in preds_shape:
        n = sh[0] * sh[1]
        fake_pred_list.append(fake_preds[idx:idx + n].reshape(sh[:-1]))
        idx += n
        
    # panel 3: region PO/predicted after changing the lab IDs
    plt.figure(figsize=(6,4))
    n_sessions = len(obs_list)
    plotted_institutions = []
    for i in range(n_sessions):
        feat = feature_list[i]
        pred = fake_pred_list[i]
        sess = sess_list[i]
        PO_idx = feat[:, 0, 0, acronym_offset+4] == 1
        lab_name = sess['session']['lab']
        session_name = sess['session']['subject']
        institution_name = institution_map[lab_name]
        institution_color = institution_colors[institution_name]

        if PO_idx.sum() == 0:
            continue

        mean_fr = np.mean(pred[PO_idx], axis=(0,1))
        baselined_mean_fr = mean_fr-mean_fr[0]
        if institution_name in plotted_institutions:
            plt.plot(baselined_mean_fr, color=institution_color, linewidth=2)
        else:
            plt.plot(baselined_mean_fr, label=institution_name, color=institution_color, linewidth=2)
        plotted_institutions.append(institution_name)
    plt.ylim(-6, 12.5)
    plt.ylabel('Baselined firing rate (sp/s)')
    plt.xlabel('Time from movement onset (s)')
    plt.axvline(10, color='k', linestyle=':')
    plt.xticks(np.arange(0, obs_list[0].shape[-1]+1, 10), labels=[-0.5, 0.0, 0.5, 1.0])
    plt.legend(ncols=1)
    plt.title('Predicted PETH of Held-out Test Trials\nAfter Fixing All Lab IDs to CCU')
    if savefig:
        figname = save_path.joinpath(f'perturbed_labID_peth.png')
        plt.savefig(figname, bbox_inches='tight', facecolor='white', dpi=600)
    plt.show()

    
def generate_figure_10_supplement4(model_config, savefig=False):
    xtick_labels = ['Original\nlab weights']
    labels = ['original']

    for mult in [1, 5, 10]:
        xtick_labels.append(f'Perturbed\nlab weights\nVarying l2 norm:\n0 - {str(7*mult)}')
        labels.append('varying_gains_factor_'+str(mult))
        
    labID_es_list = []
    for label in labels:

        load_path = save_data_path(figure='fig_mtnn').joinpath('mtnn_data')

        preds_shape = np.load(load_path.joinpath('test/shape.npy'))
        obs = np.load(load_path.joinpath(f'test/output_labID_{label}.npy'))
        test_feature = np.load(load_path.joinpath('test/feature.npy'))

        obs_list = []
        feature_list = []
        idx = 0
        for sh in preds_shape:
            n = sh[0]*sh[1]
            obs_list.append(obs[idx:idx+n].reshape(sh[:-1]))
            feature_list.append(test_feature[idx:idx+n].reshape(sh))
            idx += n

        baseline_score2 = load_test_model(model_config, ['session'], None, obs_list, preds_shape, 
                                      use_psth=False, model_name_suffix=f'labID_{label}')

        score = load_test_model(model_config, ['lab'], None, obs_list, preds_shape, 
                            use_psth=False, model_name_suffix=f'labID_{label}')

        labID_es = baseline_score2 - score

        labID_es_list.append(labID_es)
        
    np.random.seed(0)
    plt.figure(figsize=(6,6))
    plt.title('MTNN leave-one-out analysis\ncaptures simulated lab ID effects', fontsize=16)
    for i, labID_es in enumerate(labID_es_list):
        plt.scatter(0.05*np.random.normal(size=(len(labID_es)))+i*0.5, labID_es, color='k', s=24, alpha=0.1)
    plt.xticks(np.arange(len(labels))*0.5, labels=xtick_labels, rotation=0)
    plt.xlim(-0.25, len(labID_es_list)*0.5-0.25)
    plt.ylim(-0.1, 1.5)
    plt.ylabel(r'$\Delta$R2')
    if savefig:
        figname = save_path.joinpath(f'labID_perturbed.png')
        plt.savefig(figname, bbox_inches='tight', facecolor='white', dpi=600)
    plt.show()
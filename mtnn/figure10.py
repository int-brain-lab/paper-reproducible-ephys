import figrid as fg
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

from copy import deepcopy

from utils import *
from mtnn import *

from reproducible_ephys_functions import save_data_path

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
    
    load_path = save_data_path(figure='figure8').joinpath('mtnn_data')
    
    preds_shape = np.load(load_path.joinpath('test/shape.npy'))
    obs = np.load(load_path.joinpath('test/output.npy'))
    test_feature = np.load(load_path.joinpath('test/feature.npy'))
    mean_frs = compute_mean_frs()
    
    obs_list = []
    feature_list = []
    idx = 0
    for sh in preds_shape:
        n = sh[0]*sh[1]
        obs_list.append(obs[idx:idx+n].reshape(sh[:-1]))
        feature_list.append(test_feature[idx:idx+n].reshape(sh))
        idx += n
        
    region_inds = get_region_inds(feature_list)   
    baseline_score = load_test_model(model_config, None, None, obs_list, preds_shape, use_psth=use_psth)

    frs = {}
    scores = {}
    scores['leave_one_out'] = {}
    scores['single_covariate'] = {}
    scores['leave_group_out'] = {}
    for i, cov in enumerate(leave_one_out_covs):
        cov = [cov]
        scores['leave_one_out'][tuple(cov)]={}
        score = load_test_model(model_config, cov, None, obs_list, preds_shape, use_psth=use_psth)
        scores['leave_one_out'][tuple(cov)]['all'] = baseline_score-score
        for region in regions:
            region_score = np.where(region_inds==region)
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
    ax['panel_B'].set_ylim(-0.1,0.5)
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
    ax['panel_C'].set_ylim(-0.1,0.5)
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
#         ax['panel_A'].plot([i-0.3,i+0.3], [all_scores_mean,all_scores_mean], 
#                            color='k',linestyle='--')
        for j, region in enumerate(regions):
            scr = scores['single_covariate'][cov][region]
            ax['panel_A'].scatter(np.ones_like(scr)*i+0.1*(j-2), scr, 
                                  color=region_colors[region], label=region if i==0 else None, s=24, alpha=0.6)
#             ax['panel_C'].scatter(np.ones_like(scr)*i+0.1*(j-2), scr, 
#                                   c=frs[region], cmap=plt.get_cmap('Reds'), s=20, alpha=0.7)
    ax['panel_A'].legend(fontsize=24)
    ax['panel_A'].set_ylim(-0.1,0.5)
#     ax['panel_C'].set_ylim(-0.1,0.5)

    
    fg.add_labels(fig, labels)
    if savefig:
        save_path = save_data_path(figure='figure10')
        save_path.mkdir(exist_ok=True, parents=True)
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
                                           wspace=0.1, hspace=0.1)}
    
    # Add subplot labels
    labels = [{'label_text':'a', 'xpos':0, 'ypos':0, 'fontsize':30, 'weight': 'bold',
               'ha': 'right', 'va': 'bottom'},
              {'label_text':'b', 'xpos':0, 'ypos':0.37, 'fontsize':30, 'weight': 'bold',
               'ha': 'right', 'va': 'bottom'}]
    return fig, ax, labels
    
def generate_figure_10_supplement1(model_config,
                                   glm_scores,
                                   glm_leave_one_out,
                                   savefig=False):
    
    load_path = save_data_path(figure='figure10').joinpath('simulated_data')
    
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
    ax['panel_A'].set_xlim(-0.1,1.0)
    ax['panel_A'].set_ylim(-0.1,1.0)
    ax['panel_A'].plot([-1,1],[-1,1],color='k')
    ax['panel_A'].set_ylabel('GLM predictive performance (R2)')
    ax['panel_A'].set_xlabel('MTNN predictive performance (R2)')
    ax['panel_A'].set_title('GLM vs MTNN\npredictive performance on simulated data')
    
    ax['panel_B'][0][0].set_title('left stimulus')
    ax['panel_B'][0][1].set_title('right stimulus')
    ax['panel_B'][0][2].set_title('incorrect')
    ax['panel_B'][0][3].set_title('correct')
    ax['panel_B'][1][0].set_title('first movement onset')
    ax['panel_B'][1][1].set_title('mouse prior')
    ax['panel_B'][1][2].set_title('previous mouse prior')
    ax['panel_B'][1][3].set_title('wheel velocity')
    
    ax['panel_B'][0][0].set_ylabel('GLM Effect Size '+r'($\Delta$'+'R2)')
    ax['panel_B'][1][0].set_ylabel('GLM Effect Size '+r'($\Delta$'+'R2)')
    
    ax['panel_B'][1][0].set_xlabel('MTNN Effect Size '+r'($\Delta$'+'R2)')
    ax['panel_B'][1][1].set_xlabel('MTNN Effect Size '+r'($\Delta$'+'R2)')
    ax['panel_B'][1][2].set_xlabel('MTNN Effect Size '+r'($\Delta$'+'R2)')
    ax['panel_B'][1][3].set_xlabel('MTNN Effect Size '+r'($\Delta$'+'R2)')
    
    ax['panel_B'][0][0].scatter(best_score-leftstim_score, glm_leave_one_out[:,0], color='k', alpha=0.6)
    ax['panel_B'][0][0].set_xlim(-0.1,1.0)
    ax['panel_B'][0][0].set_ylim(-0.1,1.0)
    ax['panel_B'][0][0].plot([-1,1],[-1,1],color='k')
    
    ax['panel_B'][0][1].scatter(best_score-rightstim_score, glm_leave_one_out[:,1], color='k', alpha=0.6)
    ax['panel_B'][0][1].set_xlim(-0.1,1.0)
    ax['panel_B'][0][1].set_ylim(-0.1,1.0)
    ax['panel_B'][0][1].plot([-1,1],[-1,1],color='k')
    
    ax['panel_B'][0][2].scatter(best_score-incorrect_score, glm_leave_one_out[:,3], color='k', alpha=0.6)
    ax['panel_B'][0][2].set_xlim(-0.1,1.0)
    ax['panel_B'][0][2].set_ylim(-0.1,1.0)
    ax['panel_B'][0][2].plot([-1,1],[-1,1],color='k')
    
    ax['panel_B'][0][3].scatter(best_score-correct_score, glm_leave_one_out[:,2], color='k', alpha=0.6)
    ax['panel_B'][0][3].set_xlim(-0.1,1.0)
    ax['panel_B'][0][3].set_ylim(-0.1,1.0)
    ax['panel_B'][0][3].plot([-1,1],[-1,1],color='k')
    
    ax['panel_B'][1][0].scatter(best_score-fmv_score, glm_leave_one_out[:,4], color='k', alpha=0.6)
    ax['panel_B'][1][0].set_xlim(-0.1,1.0)
    ax['panel_B'][1][0].set_ylim(-0.1,1.0)
    ax['panel_B'][1][0].plot([-1,1],[-1,1],color='k')
    
    ax['panel_B'][1][1].scatter(best_score-prior_score, glm_leave_one_out[:,5], color='k', alpha=0.6)
    ax['panel_B'][1][1].set_xlim(-0.1,1.0)
    ax['panel_B'][1][1].set_ylim(-0.1,1.0)
    ax['panel_B'][1][1].plot([-1,1],[-1,1],color='k')
    
    ax['panel_B'][1][2].scatter(best_score-last_prior_score, glm_leave_one_out[:,6], color='k', alpha=0.6)
    ax['panel_B'][1][2].set_xlim(-0.1,1.0)
    ax['panel_B'][1][2].set_ylim(-0.1,1.0)
    ax['panel_B'][1][2].plot([-1,1],[-1,1],color='k')
    
    ax['panel_B'][1][3].scatter(best_score-wheel_score, glm_leave_one_out[:,7], color='k', alpha=0.6)
    ax['panel_B'][1][3].set_xlim(-0.1,1.0)
    ax['panel_B'][1][3].set_ylim(-0.1,1.0)
    ax['panel_B'][1][3].plot([-1,1],[-1,1],color='k')
    
    plt.suptitle('GLM vs MTNN Effect Sizes on Simulated Data', y=0.62)
    
    fg.add_labels(fig, labels)
    if savefig:
        save_path = save_data_path(figure='figure10')
        save_path.mkdir(exist_ok=True, parents=True)
        figname = save_path.joinpath(f'figure10_supplement1.png')
        plt.savefig(figname, bbox_inches='tight', facecolor='white')
    
    plt.show()

def generate_figure_10_supplement2(model_config,
                                   single_covs,
                                   savefig=False):
    
    color_names = ["windows blue",
                   "red",
                   "amber",
                   "faded green",
                   "dusty purple"]

    colors = sns.xkcd_palette(color_names)
    shapes = ['o', 's', '^', '+']
    
    scores, frs = compute_scores_for_figure_10(model_config,
                                              [],
                                              single_covs,
                                              [])
    
    load_path = save_data_path(figure='figure8').joinpath('mtnn_data')
    preds_shape = np.load(load_path.joinpath('test/shape.npy'))
    sess_list = np.load(load_path.joinpath('session_info.npy'), allow_pickle=True).tolist()
    
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
                axs[i,j].set_ylabel(covi, rotation=45)
                axs[i,j].yaxis.set_label_coords(-1.0, 0.5)
            if i == ncovs-1:
                axs[i,j].set_xlabel(covj, rotation=45)
                axs[i,j].xaxis.set_label_coords(0.5, -0.2)
            
            if i==0 and j==0:
                for n in range(5):
                    for m in range(4):
                        subject = sess_list[4*n+m].tolist()['session']['subject']
                        axs[i,j].scatter(-2, -2, color=colors[n], marker=shapes[m], 
                                         alpha=1.0, s=70, label=subject)
                axs[i,j].legend(bbox_to_anchor=(13.0,-5.5), fontsize=18)
                continue
            elif i == j:
                continue
                
            scorei_list = reshape_flattened(scores['single_covariate'][covi]['all'], preds_shape, trim=3)
            scorej_list = reshape_flattened(scores['single_covariate'][covj]['all'], preds_shape, trim=3)
            
            for n in range(5):
                for m in range(4):
                    axs[i,j].scatter(scorej_list[4*n+m],
                                     scorei_list[4*n+m], color=colors[n], marker=shapes[m], s=30, alpha=0.7)
            axs[i,j].plot([-1,1],[-1,1],color='black')
            axs[i,j].set_ylim(-0.1,0.55)
            axs[i,j].set_xlim(-0.1,0.55)

    plt.suptitle('Pairwise scatterplots of MTNN single-covariate effect sizes', y=0.92, fontsize=40)
    
    if savefig:
        save_path = save_data_path(figure='figure10')
        save_path.mkdir(exist_ok=True, parents=True)
        figname = save_path.joinpath(f'figure10_supplement2.png')
        plt.savefig(figname, bbox_inches='tight', facecolor='white')
    
    plt.show()
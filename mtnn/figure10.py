import figrid as fg
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

from copy import deepcopy

from utils import *
from mtnn import *

regions = list(get_acronym_dict().keys())
acronym_dict_reverse = get_acronym_dict_reverse()
region_colors = get_region_colors()

dynamic_covs = ['paw speed', 'nose speed', 'pupil diameter', 
               'motion energy', 'stimuli', 'go cue', 'first movement',
               'choice', 'reward', 'wheel velocity', 'lick', 'noise']

static_covs = ['mouse prior', 'last mouse prior', 'decision strategy (GLM-HMM)']

def make_fig_ax():
    
    xsplit = ([0.03,0.47], [0.52,1])
    ysplit = ([0.03,0.47], [0.52,1])
    
    fig = plt.figure(figsize=(30,20))
    ax = {'panel_A': fg.place_axes_on_grid(fig, xspan=xsplit[0], yspan=ysplit[0]),
          'panel_B': fg.place_axes_on_grid(fig, xspan=xsplit[1], yspan=ysplit[0]),
          'panel_C': fg.place_axes_on_grid(fig, xspan=xsplit[0], yspan=ysplit[1]),
          'panel_D': fg.place_axes_on_grid(fig, xspan=xsplit[1], yspan=ysplit[1])}
    
    # Add subplot labels
    labels = [{'label_text':'a', 'xpos':0, 'ypos':0, 'fontsize':30, 'weight': 'bold',
               'ha': 'right', 'va': 'bottom'},
              {'label_text':'b', 'xpos':0.5, 'ypos':0, 'fontsize':30, 'weight': 'bold',
               'ha': 'right', 'va': 'bottom'},
              {'label_text':'c', 'xpos':0, 'ypos':0.5, 'fontsize':30, 'weight': 'bold',
               'ha': 'right', 'va': 'bottom'},
              {'label_text':'d', 'xpos':0.5, 'ypos':0.5, 'fontsize':30, 'weight': 'bold',
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
                                 use_psth=False):
    
    preds_shape = np.load('mtnn_data/test/shape.npy')
    obs = np.load('mtnn_data/test/output.npy')
    test_feature = np.load('mtnn_data/test/feature.npy')
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
    
    model_config_dynamic = deepcopy(model_config)
    model_config_dynamic['static_bias'] = False
    baseline_score_dynamic = load_test_model(model_config_dynamic, ['all'], None, obs_list, 
                                             preds_shape, use_psth=use_psth, model_name_suffix='dynamic_bias')
    
    model_config_static = deepcopy(model_config)
    model_config_static['dynamic_bias'] = False
    baseline_score_static = load_test_model(model_config_static, ['all'], None, obs_list, 
                                            preds_shape, use_psth=use_psth, model_name_suffix='static_bias')

    frs = {}
    scores = {}
    scores['leave_one_out'] = {}
    scores['single_covariate'] = {}
    for i, cov in enumerate(leave_one_out_covs):
        scores['leave_one_out'][cov]={}
        score = load_test_model(model_config, cov, None, obs_list, preds_shape, use_psth=use_psth)
        scores['leave_one_out'][cov]['all'] = baseline_score-score
        for region in regions:
            region_score = np.where(region_inds==region)
            score_diff = baseline_score[region_score]-score[region_score]
            scores['leave_one_out'][cov][region] = score_diff
                
    for i, cov in enumerate(single_covs):
        scores['single_covariate'][cov]={}
        if cov in dynamic_covs:
            score = load_test_model(model_config_dynamic, None, cov, obs_list, 
                                    preds_shape, use_psth=use_psth)
            scores['single_covariate'][cov]['all'] = score#-baseline_score_dynamic
        else:
            score = load_test_model(model_config_static, None, cov, obs_list, 
                                    preds_shape, use_psth=use_psth)
            scores['single_covariate'][cov]['all'] = score#-baseline_score_static
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
                       savefig=False):
    
    scores, frs = compute_scores_for_figure_10(model_config,
                                          leave_one_out_covs,
                                          single_covs)

    figure_style()
    
    fig, ax, labels = make_fig_ax()
    
    ax['panel_A'].set_xticks([])
    ax['panel_B'].set_xticks([])
    ax['panel_A'].set_title('Single-covariate analysis')
    ax['panel_B'].set_title('Leave-one-out analysis')
    ax['panel_A'].set_ylabel('R2')
    ax['panel_C'].set_ylabel('R2')
    ax['panel_B'].set_ylabel(r'$\Delta$'+'R2')
    ax['panel_D'].set_ylabel(r'$\Delta$'+'R2')
    
    # Leave-one-out
    leave_one_out_covs = list(scores['leave_one_out'].keys())
    ax['panel_D'].set_xticks(np.arange(len(leave_one_out_covs)))
    ax['panel_D'].set_xticklabels(leave_one_out_covs, rotation=90, fontsize=36)
    ax['panel_B'].set_xlim(-1,len(leave_one_out_covs))
    ax['panel_D'].set_xlim(-1,len(leave_one_out_covs))
    ax['panel_B'].axhline(0, color='k', linestyle='--')
    ax['panel_D'].axhline(0, color='k', linestyle='--')
    
    for i in range(len(leave_one_out_covs)):
        cov = leave_one_out_covs[i]
        all_scores_mean = scores['leave_one_out'][cov]['all'].mean()
        ax['panel_B'].plot([i-0.3,i+0.3], [all_scores_mean,all_scores_mean], 
                           color='k',linestyle='--')
        for j, region in enumerate(regions):
            scr = scores['leave_one_out'][cov][region]
            ax['panel_B'].scatter(np.ones_like(scr)*i+0.1*(j-2), scr, 
                                  color=region_colors[region], label=region if i==0 else None, s=20, alpha=0.7)
            ax['panel_D'].scatter(np.ones_like(scr)*i+0.1*(j-2), scr, 
                                  c=frs[region], cmap=plt.get_cmap('Reds'), s=20, alpha=0.7)
    ax['panel_B'].legend(fontsize=24)
    
    # Single-covariate
    single_covs = list(scores['single_covariate'].keys())
    ax['panel_C'].set_xticks(np.arange(len(single_covs)))
    ax['panel_C'].set_xticklabels(single_covs, rotation=90, fontsize=36)
    ax['panel_A'].set_xlim(-1,len(single_covs))
    ax['panel_C'].set_xlim(-1,len(single_covs))
    ax['panel_A'].axhline(0, color='k', linestyle='--')
    ax['panel_C'].axhline(0, color='k', linestyle='--')
    
    all_scores_mean_list = []
    for i in range(len(single_covs)):
        cov = single_covs[i]
        all_scores_mean = scores['single_covariate'][cov]['all'].mean()
        all_scores_mean_list.append(all_scores_mean)
        
    single_cov_ordering = np.argsort(all_scores_mean_list)[::-1]
    for i in range(len(single_covs)):
        cov = single_covs[single_cov_ordering[i]]
        all_scores_mean = all_scores_mean_list[single_cov_ordering[i]]
        ax['panel_A'].plot([i-0.3,i+0.3], [all_scores_mean,all_scores_mean], 
                           color='k',linestyle='--')
        for j, region in enumerate(regions):
            scr = scores['single_covariate'][cov][region]
            ax['panel_A'].scatter(np.ones_like(scr)*i+0.1*(j-2), scr, 
                                  color=region_colors[region], label=region if i==0 else None, s=20, alpha=0.7)
            ax['panel_C'].scatter(np.ones_like(scr)*i+0.1*(j-2), scr, 
                                  c=frs[region], cmap=plt.get_cmap('Reds'), s=20, alpha=0.7)
    ax['panel_A'].legend(fontsize=24)
    ax['panel_A'].set_ylim(-0.1,0.5)
    ax['panel_C'].set_ylim(-0.1,0.5)
    
    
    
    fg.add_labels(fig, labels)
    if savefig:
        savedir = f'plots/figure10/'
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        figname = os.path.join(savedir, 'figure10.png')
        plt.savefig(figname, bbox_inches='tight', facecolor='white')
    
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
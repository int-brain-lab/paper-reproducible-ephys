import numpy as np
import torch
import figrid as fg
import matplotlib.pyplot as plt

from reproducible_ephys_functions import save_data_path, get_label_pos, get_row_coord, save_figure_path, figure_style, LAB_MAP
from fig_mtnn.utils import static_idx, grouped_cov_idx_dict
from fig_mtnn.mtnn import run_eval, initialize_mtnn, get_device
from fig_mtnn.fig_mtnn_plot_functions1 import (generate_figure_9, generate_figure9_supplement1, generate_figure9_supplement2,
                                               generate_figure9_supplement2_v2, generate_figure9_supplement3)

from fig_mtnn.fig_mtnn_plot_functions2 import (generate_figure_10, generate_figure_10_supplement1,
                                               generate_figure_10_supplement2, generate_figure_10_supplement3,
                                               generate_figure_10_supplement4)


data_path = save_data_path(figure='fig_mtnn')


def plot_figures9():
    data_load_path = data_path.joinpath('mtnn_data')
    feature = np.load(data_load_path.joinpath('train/feature.npy'))

    neuron_order = feature[:, 0, 0]
    feature = feature[:, :, 1:]

    neurons = np.unique(neuron_order)
    n_neurons = neurons.shape[0]
    print('number of neurons: {}'.format(n_neurons))

    INPUT_SIZE_DYNAMIC = feature.shape[-1] - static_idx.shape[0]
    INPUT_SIZE_STATIC = static_idx.shape[0]
    print(INPUT_SIZE_STATIC, INPUT_SIZE_DYNAMIC)

    HIDDEN_SIZE_STATIC = 128
    HIDDEN_SIZE_DYNAMIC = 128
    n_layers = 4

    remove_cov = None
    only_keep_cov = None

    model = initialize_mtnn(n_neurons=n_neurons,
                            input_size_static=INPUT_SIZE_STATIC,
                            input_size_dynamic=INPUT_SIZE_DYNAMIC,
                            static_bias=True, dynamic_bias=True,
                            hidden_dim_static=HIDDEN_SIZE_STATIC,
                            hidden_dim_dynamic=HIDDEN_SIZE_DYNAMIC, n_layers=n_layers,
                            dropout=0.15)

    model_load_path = data_path.joinpath(f'trained_models/state_dict_rem={remove_cov}_keep={only_keep_cov}.pt')
    model.load_state_dict(torch.load(model_load_path))

    preds, loss = run_eval(model, data_load_path.joinpath('test/feature.npy'),
                           data_load_path.joinpath('test/output.npy'),
                           remove_cov=remove_cov, only_keep_cov=only_keep_cov)
    print(f'test loss: {loss}')

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

    for i in range(len(pred_list)):
        generate_figure_9(feature_list, pred_list, obs_list,
                          neu_list, sess_list, trial_list, which_sess=[i],
                          savefig=True, plot_subsample_ratio=1.0, fr_upper_threshold=np.inf)

    model_config = {'n_neurons': n_neurons,
                    'input_size_static': INPUT_SIZE_STATIC,
                    'input_size_dynamic': INPUT_SIZE_DYNAMIC,
                    'hidden_size_static': HIDDEN_SIZE_STATIC,
                    'hidden_size_dynamic': HIDDEN_SIZE_DYNAMIC,
                    'static_bias': True,
                    'dynamic_bias': True,
                    'n_layers': n_layers}

    generate_figure9_supplement1(model_config,
                                 preds_shape,
                                 obs,
                                 test_feature,
                                 sess_list,
                                 savefig=True)

    glm_score_path = data_path.joinpath('glm_data')
    glm_score = None #np.load(glm_score_path.joinpath('glm_scores.npy'), allow_pickle=True)
    glm_score_full_mtnn_cov = np.load(glm_score_path.joinpath('glm_scores_full_mtnn_cov.npy'), allow_pickle=True)
    generate_figure9_supplement2(model_config,
                                 glm_score,
                                 glm_score_full_mtnn_cov,
                                 preds_shape,
                                 obs,
                                 test_feature,
                                 savefig=True)
    
    #generate_figure9_supplement2_v2(model_config,
    #                             glm_score,
    #                             glm_score_full_mtnn_cov,
    #                             preds_shape,
    #                             obs,
    #                             test_feature,
    #                             savefig=True)

    generate_figure9_supplement3(model_config,
                                 preds_shape,
                                 obs,
                                 test_feature,
                                 sess_list,
                                 preds,
                                 savefig=True)


def plot_supp1():

    data_load_path = data_path.joinpath('mtnn_data')
    feature = np.load(data_load_path.joinpath('train/feature.npy'))

    neuron_order = feature[:, 0, 0]
    feature = feature[:, :, 1:]

    neurons = np.unique(neuron_order)
    n_neurons = neurons.shape[0]
    print('number of neurons: {}'.format(n_neurons))

    INPUT_SIZE_DYNAMIC = feature.shape[-1] - static_idx.shape[0]
    INPUT_SIZE_STATIC = static_idx.shape[0]
    print(INPUT_SIZE_STATIC, INPUT_SIZE_DYNAMIC)

    HIDDEN_SIZE_STATIC = 128
    HIDDEN_SIZE_DYNAMIC = 128
    n_layers = 4

    remove_cov = None
    only_keep_cov = None

    model = initialize_mtnn(n_neurons=n_neurons,
                            input_size_static=INPUT_SIZE_STATIC,
                            input_size_dynamic=INPUT_SIZE_DYNAMIC,
                            static_bias=True, dynamic_bias=True,
                            hidden_dim_static=HIDDEN_SIZE_STATIC,
                            hidden_dim_dynamic=HIDDEN_SIZE_DYNAMIC, n_layers=n_layers,
                            dropout=0.15)

    model_load_path = data_path.joinpath(f'trained_models/state_dict_rem={remove_cov}_keep={only_keep_cov}.pt')
    model.load_state_dict(torch.load(model_load_path))

    preds, loss = run_eval(model, data_load_path.joinpath('test/feature.npy'),
                           data_load_path.joinpath('test/output.npy'),
                           remove_cov=remove_cov, only_keep_cov=only_keep_cov)
    print(f'test loss: {loss}')

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


    model_config = {'n_neurons': n_neurons,
                    'input_size_static': INPUT_SIZE_STATIC,
                    'input_size_dynamic': INPUT_SIZE_DYNAMIC,
                    'hidden_size_static': HIDDEN_SIZE_STATIC,
                    'hidden_size_dynamic': HIDDEN_SIZE_DYNAMIC,
                    'static_bias': True,
                    'dynamic_bias': True,
                    'n_layers': n_layers}


    width = 7
    height = 7
    figure_style()
    fig = plt.figure(figsize=(width, height))
    xspan = get_row_coord(width, [3, 2], hspace=1, pad=0.5)
    yspan = get_row_coord(height, [1, 1], hspace=1, pad=0.3)
    yspan_all = get_row_coord(height, [1], pad=0.3)

    ax = {'A': fg.place_axes_on_grid(fig, xspan=xspan[0], yspan=yspan[0], dim=[1, 2], wspace=0.05),
          'B': fg.place_axes_on_grid(fig, xspan=xspan[0], yspan=yspan[1]),
          'C': fg.place_axes_on_grid(fig, xspan=xspan[1], yspan=yspan_all[0]),
          }

    labels = [{'label_text': 'a', 'xpos': get_label_pos(width, xspan[0][0], pad=0.5),
               'ypos': get_label_pos(height, yspan[0][0], pad=0.3), 'fontsize': 10, 'weight': 'bold',
               'ha': 'right', 'va': 'bottom'},
              {'label_text': 'b', 'xpos': get_label_pos(width, xspan[0][0], pad=0.5),
               'ypos': get_label_pos(height, yspan[1][0], pad=0.3), 'fontsize': 10, 'weight': 'bold',
               'ha': 'right', 'va': 'bottom'},
              {'label_text': 'c', 'xpos': get_label_pos(width, xspan[1][0], pad=0.5),
               'ypos': get_label_pos(height, yspan_all[0][0], pad=0.3), 'fontsize': 10, 'weight': 'bold',
               'ha': 'right', 'va': 'bottom'},
              ]
    fg.add_labels(fig, labels)


    generate_figure9_supplement1(model_config,
                                 preds_shape,
                                 obs,
                                 test_feature,
                                 sess_list, axs=ax['A'],
                                 savefig=False)

    glm_score_path = data_path.joinpath('glm_data')
    glm_score = None  # np.load(glm_score_path.joinpath('glm_scores.npy'), allow_pickle=True)
    glm_score_full_mtnn_cov = np.load(glm_score_path.joinpath('glm_scores_full_mtnn_cov.npy'), allow_pickle=True)
    generate_figure9_supplement2(model_config,
                                 glm_score,
                                 glm_score_full_mtnn_cov,
                                 preds_shape,
                                 obs,
                                 test_feature, ax=ax['B'],
                                 savefig=False)

    generate_figure9_supplement3(model_config,
                                 preds_shape,
                                 obs,
                                 test_feature,
                                 sess_list,
                                 preds, ax=ax['C'],
                                 savefig=False)
    
    adjust = 0.3
    fig.subplots_adjust(top=1-(adjust + 0.2)/height, bottom=(adjust+0.2)/height, left=(adjust)/width, right=1-(adjust)/width)
    save_path = save_figure_path(figure='fig_mtnn')
    fig.savefig(save_path.joinpath('fig_mtnn_supp1.pdf'))
    fig.savefig(save_path.joinpath('fig_mtnn_supp1.png'))



def plot_supp2(test=False):
    data_load_path = data_path.joinpath('mtnn_data')

    feature = np.load(data_load_path.joinpath('train/feature.npy'))

    neuron_order = feature[:, 0, 0]
    feature = feature[:, :, 1:]

    neurons = np.unique(neuron_order)
    n_neurons = neurons.shape[0]
    print('number of neurons: {}'.format(n_neurons))

    INPUT_SIZE_DYNAMIC = feature.shape[-1] - static_idx.shape[0]
    INPUT_SIZE_STATIC = static_idx.shape[0]
    print(INPUT_SIZE_STATIC, INPUT_SIZE_DYNAMIC)

    HIDDEN_SIZE_STATIC = 128
    HIDDEN_SIZE_DYNAMIC = 128
    n_layers = 4
    if not test:
        sim_model_config = {'n_neurons': n_neurons,
                            'input_size_static': 2,
                            'input_size_dynamic': 6,
                            'hidden_size_static': HIDDEN_SIZE_STATIC,
                            'hidden_size_dynamic': HIDDEN_SIZE_DYNAMIC,
                            'static_bias': True,
                            'dynamic_bias': True,
                            'n_layers': n_layers}
    else:
        sim_model_config=None

    sim_load_path = data_path.joinpath('simulated_data')
    glm_scores = np.load(sim_load_path.joinpath('glm_scores.npy'), allow_pickle=True)
    glm_leave_one_out = np.load(sim_load_path.joinpath('glm_leave_one_out.npy'), allow_pickle=True)

    width = 7
    height = 7
    figure_style()
    fig = plt.figure(figsize=(width, height))
    xspan_row1 = get_row_coord(width, [1], span=[0.33, 0.66], pad=0)
    xspan_row2 = get_row_coord(width, [1], pad=0.5)
    yspans = get_row_coord(height, [1, 2], hspace=1.2, pad=0.3)

    ax = {'panel_A': fg.place_axes_on_grid(fig, xspan=xspan_row1[0], yspan=yspans[0]),
          'panel_B': fg.place_axes_on_grid(fig, xspan=xspan_row2[0], yspan=yspans[1], dim=[2, 4])
          }

    labels = [{'label_text': 'a', 'xpos': get_label_pos(width, xspan_row1[0][0], pad=0.5),
               'ypos': get_label_pos(height, yspans[0][0], pad=0.3), 'fontsize': 10, 'weight': 'bold',
               'ha': 'right', 'va': 'bottom'},
              {'label_text': 'b', 'xpos': get_label_pos(width, xspan_row2[0][0], pad=0.5),
               'ypos': get_label_pos(height, yspans[1][0], pad=0.3), 'fontsize': 10, 'weight': 'bold',
               'ha': 'right', 'va': 'bottom'},
              ]
    fg.add_labels(fig, labels)

    generate_figure_10_supplement1(sim_model_config, glm_scores, glm_leave_one_out, ax=ax, test=test, savefig=False)
    adjust = 0.3
    fig.subplots_adjust(top=1-adjust/height, bottom=(adjust + 0.2)/height, left=(adjust)/width, right=1-(adjust)/width)
    save_path = save_figure_path(figure='fig_mtnn')
    fig.savefig(save_path.joinpath('fig_mtnn_supp2.pdf'))
    fig.savefig(save_path.joinpath('fig_mtnn_supp2.png'))



def plot_supp4(test=False):

    data_load_path = data_path.joinpath('mtnn_data')

    feature = np.load(data_load_path.joinpath('train/feature.npy'))

    neuron_order = feature[:, 0, 0]
    feature = feature[:, :, 1:]

    neurons = np.unique(neuron_order)
    n_neurons = neurons.shape[0]
    print('number of neurons: {}'.format(n_neurons))

    INPUT_SIZE_DYNAMIC = feature.shape[-1] - static_idx.shape[0]
    INPUT_SIZE_STATIC = static_idx.shape[0]
    print(INPUT_SIZE_STATIC, INPUT_SIZE_DYNAMIC)

    HIDDEN_SIZE_STATIC = 128
    HIDDEN_SIZE_DYNAMIC = 128
    n_layers = 4

    model_config = {'n_neurons': n_neurons,
                    'input_size_static': INPUT_SIZE_STATIC,
                    'input_size_dynamic': INPUT_SIZE_DYNAMIC,
                    'hidden_size_static': HIDDEN_SIZE_STATIC,
                    'hidden_size_dynamic': HIDDEN_SIZE_DYNAMIC,
                    'static_bias': True,
                    'dynamic_bias': True,
                    'n_layers': n_layers}


    single_covs_supplement2 = ['paw speed', 'nose speed', 'pupil diameter',
                               'motion energy', 'stimuli', 'go cue', 'first movement',
                               'choice', 'reward', 'wheel velocity', 'lick']

    ncovs = len(single_covs_supplement2)
    figure_style()
    width = 7
    height = 7
    fig, axs = plt.subplots(ncovs, ncovs, figsize=(width,height), sharey=True, sharex=True)
    #plt.subplots_adjust(wspace=0.03, hspace=0.03)
    generate_figure_10_supplement2(model_config, single_covs_supplement2, axs=axs, savefig=False)
    # adjust = 0.1
    # fig.subplots_adjust(left=0.01)
    save_path = save_figure_path(figure='fig_mtnn')
    fig.savefig(save_path.joinpath('fig_mtnn_supp4.pdf'))
    fig.savefig(save_path.joinpath('fig_mtnn_supp4.png'))



def plot_supp3(test=False):

    data_load_path = data_path.joinpath('mtnn_data')

    feature = np.load(data_load_path.joinpath('train/feature.npy'))

    neuron_order = feature[:, 0, 0]
    feature = feature[:, :, 1:]

    neurons = np.unique(neuron_order)
    n_neurons = neurons.shape[0]
    print('number of neurons: {}'.format(n_neurons))

    INPUT_SIZE_DYNAMIC = feature.shape[-1] - static_idx.shape[0]
    INPUT_SIZE_STATIC = static_idx.shape[0]
    print(INPUT_SIZE_STATIC, INPUT_SIZE_DYNAMIC)

    HIDDEN_SIZE_STATIC = 128
    HIDDEN_SIZE_DYNAMIC = 128
    n_layers = 4

    if not test:
        model_config = {'n_neurons': n_neurons,
                        'input_size_static': INPUT_SIZE_STATIC,
                        'input_size_dynamic': INPUT_SIZE_DYNAMIC,
                        'hidden_size_static': HIDDEN_SIZE_STATIC,
                        'hidden_size_dynamic': HIDDEN_SIZE_DYNAMIC,
                        'static_bias': True,
                        'dynamic_bias': True,
                        'n_layers': n_layers}
    else:
        model_config = None


    width = 7
    height = 7
    figure_style()
    fig = plt.figure(figsize=(width, height))
    xspan_row1 = get_row_coord(width, [1, 1, 1], pad=0.5)
    xspan_row2 = get_row_coord(width, [1], pad=0.5)
    xspan_row3 = get_row_coord(width, [1], span=[0.2, 0.8], pad=0)
    yspans = get_row_coord(height, [4, 1, 5], hspace=[0.1, 0.8], pad=0.3)

    ax = {'A_1': fg.place_axes_on_grid(fig, xspan=xspan_row1[0], yspan=yspans[0]),
          'A_2': fg.place_axes_on_grid(fig, xspan=xspan_row1[1], yspan=yspans[0]),
          'A_3': fg.place_axes_on_grid(fig, xspan=xspan_row1[2], yspan=yspans[0]),
          'labs': fg.place_axes_on_grid(fig, xspan=xspan_row2[0], yspan=yspans[1]),
          'B': fg.place_axes_on_grid(fig, xspan=xspan_row3[0], yspan=yspans[2]),
          }

    labels = [{'label_text': 'a', 'xpos': get_label_pos(width, xspan_row1[0][0], pad=0.5),
               'ypos': get_label_pos(height, yspans[0][0], pad=0.3), 'fontsize': 10, 'weight': 'bold',
               'ha': 'right', 'va': 'bottom'},
              {'label_text': 'b', 'xpos': get_label_pos(width, xspan_row3[0][0], pad=0.5),
               'ypos': get_label_pos(height, yspans[2][0], pad=0.3), 'fontsize': 10, 'weight': 'bold',
               'ha': 'right', 'va': 'bottom'},
              ]
    fg.add_labels(fig, labels)

    inst = generate_figure_10_supplement3(model_config, test=test, axs=[ax['A_1'], ax['A_2'], ax['A_3']], savefig=False)

    generate_figure_10_supplement4(model_config, test=test, ax=ax['B'], savefig=False)

    ax['labs'].set_axis_off()
    _, _, institution_colors = LAB_MAP()
    inst = list(set(inst))
    inst.sort()
    for i, l in enumerate(inst):
        if i == 0:
            text = ax['labs'].text(0.3, 0, l, va='bottom', color=institution_colors[l], fontsize=8, transform=ax['labs'].transAxes)
        else:
            text = ax['labs'].annotate(
                '  ' + l, xycoords=text, xy=(1, 0), verticalalignment="bottom",
                color=institution_colors[l], fontsize=8)  # custom properties

    adjust = 0.3
    fig.subplots_adjust(top=1-adjust/height, bottom=(adjust + 0.2)/height, left=(adjust)/width,
                        right=1-(adjust)/width)
    save_path = save_figure_path(figure='fig_mtnn')
    fig.savefig(save_path.joinpath('fig_mtnn_supp3.pdf'))
    fig.savefig(save_path.joinpath('fig_mtnn_supp3.png'))





def plot_figures10():
    data_load_path = data_path.joinpath('mtnn_data')

    feature = np.load(data_load_path.joinpath('train/feature.npy'))

    neuron_order = feature[:, 0, 0]
    feature = feature[:, :, 1:]

    neurons = np.unique(neuron_order)
    n_neurons = neurons.shape[0]
    print('number of neurons: {}'.format(n_neurons))

    INPUT_SIZE_DYNAMIC = feature.shape[-1] - static_idx.shape[0]
    INPUT_SIZE_STATIC = static_idx.shape[0]
    print(INPUT_SIZE_STATIC, INPUT_SIZE_DYNAMIC)

    HIDDEN_SIZE_STATIC = 128
    HIDDEN_SIZE_DYNAMIC = 128
    n_layers = 4


    model_config = {'n_neurons': n_neurons,
                    'input_size_static': INPUT_SIZE_STATIC,
                    'input_size_dynamic': INPUT_SIZE_DYNAMIC,
                    'hidden_size_static': HIDDEN_SIZE_STATIC,
                    'hidden_size_dynamic': HIDDEN_SIZE_DYNAMIC,
                    'static_bias': True,
                    'dynamic_bias': True,
                    'n_layers': n_layers}

    leave_one_out_covs = ['lab', 'session', 'x', 'y', 'z',
                          'waveform amplitude', 'waveform width',
                          'paw speed', 'nose speed', 'pupil diameter', 'motion energy',
                          'stimuli', 'go cue', 'first movement', 'choice',
                          'reward', 'wheel velocity', 'mouse prior', 'last mouse prior',
                          'lick', 'decision strategy (GLM-HMM)', 'brain region', 'noise']
    leave_group_out = [grouped_cov_idx_dict['ephys'], grouped_cov_idx_dict['task'],
                       grouped_cov_idx_dict['behavioral']]
    single_covs = ['paw speed', 'nose speed', 'pupil diameter',
                   'motion energy', 'stimuli', 'go cue', 'first movement',
                   'choice', 'reward', 'wheel velocity', 'lick', 'noise',
                   'mouse prior', 'last mouse prior', 'decision strategy (GLM-HMM)']

#     leave_one_out_covs = ['lab', 'session', 'x', 'y']
#     leave_group_out = []
#     single_covs = ['paw speed', 'nose speed', 'pupil diameter',
#                    'motion energy', 'stimuli']

    generate_figure_10(model_config, leave_one_out_covs, single_covs, leave_group_out, savefig=True)

    sim_model_config = {'n_neurons': n_neurons,
                        'input_size_static': 2,
                        'input_size_dynamic': 6,
                        'hidden_size_static': HIDDEN_SIZE_STATIC,
                        'hidden_size_dynamic': HIDDEN_SIZE_DYNAMIC,
                        'static_bias': True,
                        'dynamic_bias': True,
                        'n_layers': n_layers}

    sim_load_path = data_path.joinpath('simulated_data')
    glm_scores = np.load(sim_load_path.joinpath('glm_scores.npy'), allow_pickle=True)
    glm_leave_one_out = np.load(sim_load_path.joinpath('glm_leave_one_out.npy'), allow_pickle=True)
    generate_figure_10_supplement1(sim_model_config, glm_scores, glm_leave_one_out, savefig=True)

    single_covs_supplement2 = ['paw speed', 'nose speed', 'pupil diameter',
                               'motion energy', 'stimuli', 'go cue', 'first movement',
                               'choice', 'reward', 'wheel velocity', 'lick']
    generate_figure_10_supplement2(model_config, single_covs_supplement2, savefig=True)
    
    generate_figure_10_supplement3(model_config, savefig=True)
    
    generate_figure_10_supplement4(model_config, savefig=True)


if __name__ == '__main__':
    plot_figures9()
    plot_figures10()
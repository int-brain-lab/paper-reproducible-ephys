import numpy as np
import torch

from reproducible_ephys_functions import save_data_path
from fig_mtnn.utils import static_idx, grouped_cov_idx_dict
from fig_mtnn.mtnn import run_eval, initialize_mtnn, get_device
from fig_mtnn.fig_mtnn_plot_functions1 import (generate_figure_9, generate_figure9_supplement1, generate_figure9_supplement2,
                                                            generate_figure9_supplement2_v2, generate_figure9_supplement3)

from fig_mtnn.fig_mtnn_plot_functions2 import (generate_figure_10, generate_figure_10_supplement1,
                                                            generate_figure_10_supplement2)


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

    leave_one_out_covs = ['lab', 'session', 'x', 'y']
    leave_group_out = []
    single_covs = ['paw speed', 'nose speed', 'pupil diameter',
                   'motion energy', 'stimuli']

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
import numpy as np

from reproducible_ephys_functions import save_data_path
from figure9_10.utils import static_idx, grouped_cov_idx_dict
from figure9_10.figure10_plot_functions import generate_figure_10, generate_figure_10_supplement1, generate_figure_10_supplement2

data_path = save_data_path(figure='figure9_10')

def plot_figures():
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

    HIDDEN_SIZE_STATIC = 64
    HIDDEN_SIZE_DYNAMIC = 64
    n_layers = 3


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

    generate_figure_10_supplement1(sim_model_config,
                                   glm_scores,
                                   glm_leave_one_out,
                                   savefig=True)

    single_covs_supplement2 = ['paw speed', 'nose speed', 'pupil diameter',
                               'motion energy', 'stimuli', 'go cue', 'first movement',
                               'choice', 'reward', 'wheel velocity', 'lick']
    generate_figure_10_supplement2(model_config, single_covs_supplement2, savefig=True)

if __name__ == '__main__':
    plot_figures()

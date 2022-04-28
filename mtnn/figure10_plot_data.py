from utils import *
from mtnn import *
from figure10 import *
import torch
import matplotlib.pyplot as plt
import numpy as np
from reproducible_ephys_functions import save_data_path

data_load_path = save_data_path(figure='figure8').joinpath('mtnn_data')
sim_data_load_path = save_data_path(figure='figure10').joinpath('simulated_data')

feature = np.load(data_load_path.joinpath('train/feature.npy'))

neuron_order = feature[:,0,0]
feature = feature[:,:,1:]

neurons = np.unique(neuron_order)
n_neurons = neurons.shape[0]
print('number of neurons: {}'.format(n_neurons))

INPUT_SIZE_DYNAMIC = feature.shape[-1]-static_idx.shape[0]
INPUT_SIZE_STATIC = static_idx.shape[0]
print(INPUT_SIZE_STATIC, INPUT_SIZE_DYNAMIC)

HIDDEN_SIZE_STATIC = 64
HIDDEN_SIZE_DYNAMIC = 64
n_layers = 2

do_train = True
sim_do_train = True
n_epochs=100

if do_train:
    # train leave-one-out
    for i, key in notebook.tqdm(enumerate(cov_idx_dict.keys())):
        print(f'processing {key}')

        remove_cov = [key]
        only_keep_cov = None

        model = initialize_mtnn(n_neurons=n_neurons, 
                                input_size_static=INPUT_SIZE_STATIC, 
                                input_size_dynamic=INPUT_SIZE_DYNAMIC,
                                static_bias=True, dynamic_bias=True, 
                                hidden_dim_static=HIDDEN_SIZE_STATIC, 
                                hidden_dim_dynamic=HIDDEN_SIZE_DYNAMIC, n_layers=n_layers, 
                                dropout=0.2)

        best_epoch, loss_list, val_loss_list = run_train(model, 
                                                         data_load_path.joinpath('train/feature.npy'), 
                                                         data_load_path.joinpath('train/output.npy'), 
                                                         data_load_path.joinpath('validation/feature.npy'), 
                                                         data_load_path.joinpath('validation/output.npy'),
                                                         batch_size=512, n_epochs=n_epochs, lr=0.1,
                                                         weight_decay=1e-5,
                                                         remove_cov=remove_cov, 
                                                         only_keep_cov=only_keep_cov)
        
    # train dynamic single-covariate
    for i, key in notebook.tqdm(enumerate(cov_idx_dict.keys())):

        if key not in ['paw speed', 'nose speed', 'pupil diameter', 'motion energy',
                       'stimuli', 'go cue', 'first movement', 'choice', 'reward',
                       'wheel velocity', 'lick', 'noise']:
            continue

        print(f'processing {key}')

        remove_cov = None
        only_keep_cov = key

        model = initialize_mtnn(n_neurons=n_neurons, 
                                input_size_static=INPUT_SIZE_STATIC, 
                                input_size_dynamic=INPUT_SIZE_DYNAMIC,
                                static_bias=False, dynamic_bias=True, 
                                hidden_dim_static=HIDDEN_SIZE_STATIC, 
                                hidden_dim_dynamic=HIDDEN_SIZE_DYNAMIC, n_layers=n_layers, 
                                dropout=0.2)

        best_epoch, loss_list, val_loss_list = run_train(model, 
                                                         data_load_path.joinpath('train/feature.npy'), 
                                                         data_load_path.joinpath('train/output.npy'), 
                                                         data_load_path.joinpath('validation/feature.npy'), 
                                                         data_load_path.joinpath('validation/output.npy'),
                                                         batch_size=512, n_epochs=n_epochs, lr=0.1,
                                                         weight_decay=1e-5,
                                                         remove_cov=remove_cov, 
                                                         only_keep_cov=only_keep_cov)
        
    # train static single-covariate
    for i, key in notebook.tqdm(enumerate(cov_idx_dict.keys())):

        if key in ['paw speed', 'nose speed', 'pupil diameter', 'motion energy',
                   'stimuli', 'go cue', 'first movement', 'choice', 'reward',
                   'wheel velocity', 'lick', 'noise', 'all']:
            continue

        print(f'processing {key}')

        remove_cov = None
        only_keep_cov = key

        model = initialize_mtnn(n_neurons=n_neurons, 
                                input_size_static=INPUT_SIZE_STATIC, 
                                input_size_dynamic=INPUT_SIZE_DYNAMIC,
                                static_bias=True, dynamic_bias=False, 
                                hidden_dim_static=HIDDEN_SIZE_STATIC, 
                                hidden_dim_dynamic=HIDDEN_SIZE_DYNAMIC, n_layers=n_layers, 
                                dropout=0.2)

        best_epoch, loss_list, val_loss_list = run_train(model, 
                                                         data_load_path.joinpath('train/feature.npy'), 
                                                         data_load_path.joinpath('train/output.npy'), 
                                                         data_load_path.joinpath('validation/feature.npy'), 
                                                         data_load_path.joinpath('validation/output.npy'),
                                                         batch_size=512, n_epochs=n_epochs, lr=0.1,
                                                         weight_decay=1e-5,
                                                         remove_cov=remove_cov, 
                                                         only_keep_cov=only_keep_cov)
        
    # train leave-group-out
    for i, key in notebook.tqdm(enumerate(grouped_cov_idx_dict.keys())):
        print(f'processing {key}')

        remove_cov = grouped_cov_idx_dict[key]
        only_keep_cov = None

        model = initialize_mtnn(n_neurons=n_neurons, 
                                input_size_static=INPUT_SIZE_STATIC, 
                                input_size_dynamic=INPUT_SIZE_DYNAMIC,
                                static_bias=True, dynamic_bias=True, 
                                hidden_dim_static=HIDDEN_SIZE_STATIC, 
                                hidden_dim_dynamic=HIDDEN_SIZE_DYNAMIC, n_layers=n_layers, 
                                dropout=0.2)

        best_epoch, loss_list, val_loss_list = run_train(model, 
                                                         data_load_path.joinpath('train/feature.npy'), 
                                                         data_load_path.joinpath('train/output.npy'), 
                                                         data_load_path.joinpath('validation/feature.npy'), 
                                                         data_load_path.joinpath('validation/output.npy'),
                                                         batch_size=512, n_epochs=n_epochs, lr=0.1,
                                                         weight_decay=1e-5,
                                                         remove_cov=remove_cov, 
                                                         only_keep_cov=only_keep_cov)
        
model_config = {'n_neurons': n_neurons,
                'input_size_static': INPUT_SIZE_STATIC,
                'input_size_dynamic': INPUT_SIZE_DYNAMIC,
                'hidden_size_static': HIDDEN_SIZE_STATIC,
                'hidden_size_dynamic': HIDDEN_SIZE_DYNAMIC,
                'static_bias': True,
                'dynamic_bias': True,
                'n_layers': n_layers}

leave_one_out_covs = ['lab','session', 'x', 'y', 'z', 
                      'waveform amplitude', 'waveform width', 
                      'paw speed', 'nose speed', 'pupil diameter', 'motion energy', 
                      'stimuli','go cue','first movement','choice',
                      'reward','wheel velocity', 'mouse prior', 'last mouse prior',
                      'lick','decision strategy (GLM-HMM)','brain region','noise']
leave_group_out=[grouped_cov_idx_dict['ephys'], grouped_cov_idx_dict['task'], 
                      grouped_cov_idx_dict['behavioral']]
single_covs = ['paw speed', 'nose speed', 'pupil diameter', 
               'motion energy', 'stimuli', 'go cue', 'first movement',
               'choice', 'reward', 'wheel velocity', 'lick', 'noise',
               'mouse prior', 'last mouse prior', 'decision strategy (GLM-HMM)']

generate_figure_10(model_config, leave_one_out_covs, single_covs, leave_group_out, savefig=True)

if sim_do_train:

    model = initialize_mtnn(n_neurons=n_neurons, 
                            input_size_static=2, 
                            input_size_dynamic=6,
                            static_bias=True, dynamic_bias=True, 
                            hidden_dim_static=HIDDEN_SIZE_STATIC, 
                            hidden_dim_dynamic=HIDDEN_SIZE_DYNAMIC, n_layers=n_layers, 
                            dropout=0.2)

    best_epoch, loss_list, val_loss_list = run_train(model, 
                                                     sim_data_load_path.joinpath('train/feature.npy'), 
                                                     sim_data_load_path.joinpath('train/output.npy'), 
                                                     sim_data_load_path.joinpath('validation/feature.npy'), 
                                                     sim_data_load_path.joinpath('validation/output.npy'),
                                                     batch_size=512, n_epochs=n_epochs, lr=0.1,
                                                     weight_decay=1e-5,
                                                     remove_cov=None, 
                                                     only_keep_cov=None, simulated=True)
    
    # train simulated leave-group-out
    for i, key in notebook.tqdm(enumerate(sim_cov_idx_dict.keys())):
        print(f'processing {key}')

        remove_cov = [key]
        only_keep_cov = None

        model = initialize_mtnn(n_neurons=n_neurons, 
                                input_size_static=2, 
                                input_size_dynamic=6,
                                static_bias=True, dynamic_bias=True, 
                                hidden_dim_static=HIDDEN_SIZE_STATIC, 
                                hidden_dim_dynamic=HIDDEN_SIZE_DYNAMIC, n_layers=n_layers, 
                                dropout=0.2)

        best_epoch, loss_list, val_loss_list = run_train(model, 
                                                         sim_data_load_path.joinpath('train/feature.npy'), 
                                                         sim_data_load_path.joinpath('train/output.npy'), 
                                                         sim_data_load_path.joinpath('validation/feature.npy'), 
                                                         sim_data_load_path.joinpath('validation/output.npy'),
                                                         batch_size=512, n_epochs=n_epochs, lr=0.1,
                                                         weight_decay=1e-5,
                                                         remove_cov=remove_cov, 
                                                         only_keep_cov=only_keep_cov, simulated=True)
    

sim_model_config = {'n_neurons': n_neurons,
                    'input_size_static': 2,
                    'input_size_dynamic': 6,
                    'hidden_size_static': HIDDEN_SIZE_STATIC,
                    'hidden_size_dynamic': HIDDEN_SIZE_DYNAMIC,
                    'static_bias': True,
                    'dynamic_bias': True,
                    'n_layers': n_layers}

sim_load_path = save_data_path(figure='figure10').joinpath('simulated_data')
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
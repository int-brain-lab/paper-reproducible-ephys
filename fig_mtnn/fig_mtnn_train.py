from fig_mtnn.utils import *
from fig_mtnn.mtnn import run_eval, initialize_mtnn, get_device, run_train, load_test_model
import numpy as np
from reproducible_ephys_functions import save_data_path
from tqdm import notebook
import torch

def train_all(do_train=True, do_train_on_glm_covariates=True):

    data_load_path = save_data_path(figure='fig_mtnn').joinpath('mtnn_data')
    feature = np.load(data_load_path.joinpath('train/feature.npy'))

    neuron_order = feature[:, 0, 0]
    feature = feature[:, :, 1:]

    neurons = np.unique(neuron_order)
    n_neurons = neurons.shape[0]
    print('number of neurons: {}'.format(n_neurons))

    INPUT_SIZE_DYNAMIC = feature.shape[-1]-static_idx.shape[0]
    INPUT_SIZE_STATIC = static_idx.shape[0]
    print(INPUT_SIZE_STATIC, INPUT_SIZE_DYNAMIC)

    HIDDEN_SIZE_STATIC = 128
    HIDDEN_SIZE_DYNAMIC = 128
    n_layers = 4
    weight_decay = 1e-15 #1e-8
    dropout = 0.15 #0.2
    batch_size = 1024 #512
    lr = 0.1

    model = initialize_mtnn(n_neurons=n_neurons,
                            input_size_static=INPUT_SIZE_STATIC,
                            input_size_dynamic=INPUT_SIZE_DYNAMIC,
                            static_bias=True, dynamic_bias=True,
                            hidden_dim_static=HIDDEN_SIZE_STATIC,
                            hidden_dim_dynamic=HIDDEN_SIZE_DYNAMIC, n_layers=n_layers,
                            dropout=dropout)

    if do_train:
        best_epoch, loss_list, val_loss_list = run_train(model,
                                                         data_load_path.joinpath('train/feature.npy'),
                                                         data_load_path.joinpath('train/output.npy'),
                                                         data_load_path.joinpath('validation/feature.npy'),
                                                         data_load_path.joinpath('validation/output.npy'),
                                                         batch_size=batch_size, n_epochs=n_epochs, lr=lr,
                                                         weight_decay=weight_decay,
                                                         remove_cov=remove_cov,
                                                         only_keep_cov=only_keep_cov)

    if do_train_on_glm_covariates:
        model = initialize_mtnn(n_neurons=n_neurons,
                                input_size_static=INPUT_SIZE_STATIC,
                                input_size_dynamic=INPUT_SIZE_DYNAMIC,
                                static_bias=True, dynamic_bias=True,
                                hidden_dim_static=HIDDEN_SIZE_STATIC,
                                hidden_dim_dynamic=HIDDEN_SIZE_DYNAMIC, n_layers=n_layers,
                                dropout=dropout)

        best_epoch, loss_list, val_loss_list = run_train(model,
                                                         data_load_path.joinpath('train/feature.npy'),
                                                         data_load_path.joinpath('train/output.npy'),
                                                         data_load_path.joinpath('validation/feature.npy'),
                                                         data_load_path.joinpath('validation/output.npy'),
                                                         batch_size=batch_size, n_epochs=n_epochs, lr=lr,
                                                         weight_decay=weight_decay,
                                                         remove_cov=leave_out_covs_for_glm,
                                                         only_keep_cov=only_keep_cov)


def train_groups(do_train=True, sim_do_train=True):
    data_load_path = save_data_path(figure='fig_mtnn').joinpath('mtnn_data')
    sim_data_load_path = save_data_path(figure='fig_mtnn').joinpath('simulated_data')

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
    weight_decay = 1e-15 #1e-8
    dropout = 0.15 #0.2
    batch_size = 1024#512
    lr = 0.1

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
                                    dropout=dropout)

            best_epoch, loss_list, val_loss_list = run_train(model,
                                                             data_load_path.joinpath('train/feature.npy'),
                                                             data_load_path.joinpath('train/output.npy'),
                                                             data_load_path.joinpath('validation/feature.npy'),
                                                             data_load_path.joinpath('validation/output.npy'),
                                                             batch_size=batch_size, n_epochs=n_epochs, lr=lr,
                                                             weight_decay=weight_decay,
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
                                    dropout=dropout)

            best_epoch, loss_list, val_loss_list = run_train(model,
                                                             data_load_path.joinpath('train/feature.npy'),
                                                             data_load_path.joinpath('train/output.npy'),
                                                             data_load_path.joinpath('validation/feature.npy'),
                                                             data_load_path.joinpath('validation/output.npy'),
                                                             batch_size=batch_size, n_epochs=n_epochs, lr=lr,
                                                             weight_decay=weight_decay,
                                                             remove_cov=remove_cov,
                                                             only_keep_cov=only_keep_cov)

        # train static single-covariate
        for i, key in notebook.tqdm(enumerate(cov_idx_dict.keys())):

            if key not in ['mouse prior', 'last mouse prior', 'decision strategy (GLM-HMM)']:
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
                                    dropout=dropout)

            best_epoch, loss_list, val_loss_list = run_train(model,
                                                             data_load_path.joinpath('train/feature.npy'),
                                                             data_load_path.joinpath('train/output.npy'),
                                                             data_load_path.joinpath('validation/feature.npy'),
                                                             data_load_path.joinpath('validation/output.npy'),
                                                             batch_size=batch_size, n_epochs=n_epochs, lr=lr,
                                                             weight_decay=weight_decay,
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
                                    dropout=dropout)

            best_epoch, loss_list, val_loss_list = run_train(model,
                                                             data_load_path.joinpath('train/feature.npy'),
                                                             data_load_path.joinpath('train/output.npy'),
                                                             data_load_path.joinpath('validation/feature.npy'),
                                                             data_load_path.joinpath('validation/output.npy'),
                                                             batch_size=batch_size, n_epochs=n_epochs, lr=lr,
                                                             weight_decay=weight_decay,
                                                             remove_cov=remove_cov,
                                                             only_keep_cov=only_keep_cov)

    if sim_do_train:

        model = initialize_mtnn(n_neurons=n_neurons,
                                input_size_static=2,
                                input_size_dynamic=6,
                                static_bias=True, dynamic_bias=True,
                                hidden_dim_static=HIDDEN_SIZE_STATIC,
                                hidden_dim_dynamic=HIDDEN_SIZE_DYNAMIC, n_layers=n_layers,
                                dropout=dropout)

        best_epoch, loss_list, val_loss_list = run_train(model,
                                                         sim_data_load_path.joinpath('train/feature.npy'),
                                                         sim_data_load_path.joinpath('train/output.npy'),
                                                         sim_data_load_path.joinpath('validation/feature.npy'),
                                                         sim_data_load_path.joinpath('validation/output.npy'),
                                                         batch_size=batch_size, n_epochs=n_epochs, lr=lr,
                                                         weight_decay=weight_decay,
                                                         remove_cov=None,
                                                         only_keep_cov=None, simulated=True)

        # train simulated leave-one-out
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
                                    dropout=dropout)

            best_epoch, loss_list, val_loss_list = run_train(model,
                                                             sim_data_load_path.joinpath('train/feature.npy'),
                                                             sim_data_load_path.joinpath('train/output.npy'),
                                                             sim_data_load_path.joinpath('validation/feature.npy'),
                                                             sim_data_load_path.joinpath('validation/output.npy'),
                                                             batch_size=batch_size, n_epochs=n_epochs, lr=lr,
                                                             weight_decay=weight_decay,
                                                             remove_cov=remove_cov,
                                                             only_keep_cov=only_keep_cov, simulated=True)


def train_labID_exp():
    data_path = save_data_path(figure='fig_mtnn')
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
    weight_decay = 1e-15
    dropout = 0.15
    batch_size = 1024
    lr = 0.1
    
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
    model.eval()
    
    device = get_device()

    # lab weights
    lab_weights_list = [model.fc_static1.weight[:, lab_offset-1:session_offset-1].cpu().detach().numpy()]
    labels = ['original']

    np.random.seed(0)
    lab_orthog_weights = np.linalg.qr(np.random.normal(size=(128,128)))[0][:, :8]

    for mult in [1, 5, 10]:
        lab_weights_list.append(lab_orthog_weights * (np.arange(0, 8)[None] * mult))
        labels.append('varying_gains_factor_'+str(mult))

    # generate data
    with torch.no_grad():
        for lab_weight, label in zip(lab_weights_list, labels):
            model = initialize_mtnn(n_neurons=n_neurons,
                                    input_size_static=INPUT_SIZE_STATIC,
                                    input_size_dynamic=INPUT_SIZE_DYNAMIC,
                                    static_bias=True, dynamic_bias=True,
                                    hidden_dim_static=HIDDEN_SIZE_STATIC,
                                    hidden_dim_dynamic=HIDDEN_SIZE_DYNAMIC, n_layers=n_layers,
                                    dropout=dropout)

            model_load_path = data_path.joinpath(f'trained_models/state_dict_rem={remove_cov}_keep={only_keep_cov}.pt')
            model.load_state_dict(torch.load(model_load_path))
            model.eval()

            model.fc_static1.weight[:, lab_offset-1:session_offset-1] = torch.Tensor(lab_weight).to(device)

            train_new_preds, loss = run_eval(model, data_load_path.joinpath('train/feature.npy'),
                                           data_load_path.joinpath('train/output.npy'),
                                           remove_cov=remove_cov, only_keep_cov=only_keep_cov)

            val_new_preds, loss = run_eval(model, data_load_path.joinpath('validation/feature.npy'),
                                           data_load_path.joinpath('validation/output.npy'),
                                           remove_cov=remove_cov, only_keep_cov=only_keep_cov)

            test_new_preds, loss = run_eval(model, data_load_path.joinpath('test/feature.npy'),
                                           data_load_path.joinpath('test/output.npy'),
                                           remove_cov=remove_cov, only_keep_cov=only_keep_cov)

            np.save(data_load_path.joinpath(f'train/output_labID_{label}.npy'), train_new_preds)
            np.save(data_load_path.joinpath(f'validation/output_labID_{label}.npy'), val_new_preds)
            np.save(data_load_path.joinpath(f'test/output_labID_{label}.npy'), test_new_preds)

            
    # start actual training
    feature = np.load(data_load_path.joinpath('train/feature.npy'))

    neuron_order = feature[:, 0, 0]
    feature = feature[:, :, 1:]

    neurons = np.unique(neuron_order)
    n_neurons = neurons.shape[0]
    print('number of neurons: {}'.format(n_neurons))

    INPUT_SIZE_DYNAMIC = feature.shape[-1]-static_idx.shape[0]
    INPUT_SIZE_STATIC = static_idx.shape[0]
    print(INPUT_SIZE_STATIC, INPUT_SIZE_DYNAMIC)

    HIDDEN_SIZE_STATIC = 128
    HIDDEN_SIZE_DYNAMIC = 128
    n_layers = 4
    weight_decay = 0.0
    dropout = 0.15
    batch_size = 1024
    lr = 0.1

    n_epochs = 100
    only_keep_cov = None

    for label in labels:
        print(label)

        model = initialize_mtnn(n_neurons=n_neurons,
                                input_size_static=INPUT_SIZE_STATIC,
                                input_size_dynamic=INPUT_SIZE_DYNAMIC,
                                static_bias=True, dynamic_bias=True,
                                hidden_dim_static=HIDDEN_SIZE_STATIC,
                                hidden_dim_dynamic=HIDDEN_SIZE_DYNAMIC, n_layers=n_layers,
                                dropout=dropout)

        best_epoch, loss_list, val_loss_list = run_train(model,
                                                         data_load_path.joinpath('train/feature.npy'),
                                                         data_load_path.joinpath(f'train/output_labID_{label}.npy'),
                                                         data_load_path.joinpath('validation/feature.npy'),
                                                         data_load_path.joinpath(f'validation/output_labID_{label}.npy'),
                                                         batch_size=batch_size, n_epochs=n_epochs, lr=lr,
                                                         weight_decay=weight_decay,
                                                         remove_cov=['session'],
                                                         only_keep_cov=only_keep_cov,
                                                         model_name_suffix=f'labID_{label}')

        model = initialize_mtnn(n_neurons=n_neurons,
                                input_size_static=INPUT_SIZE_STATIC,
                                input_size_dynamic=INPUT_SIZE_DYNAMIC,
                                static_bias=True, dynamic_bias=True,
                                hidden_dim_static=HIDDEN_SIZE_STATIC,
                                hidden_dim_dynamic=HIDDEN_SIZE_DYNAMIC, n_layers=n_layers,
                                dropout=dropout)

        best_epoch, loss_list, val_loss_list = run_train(model,
                                                         data_load_path.joinpath('train/feature.npy'),
                                                         data_load_path.joinpath(f'train/output_labID_{label}.npy'),
                                                         data_load_path.joinpath('validation/feature.npy'),
                                                         data_load_path.joinpath(f'validation/output_labID_{label}.npy'),
                                                         batch_size=batch_size, n_epochs=n_epochs, lr=lr,
                                                         weight_decay=weight_decay,
                                                         remove_cov=['lab'],
                                                         only_keep_cov=only_keep_cov,
                                                         model_name_suffix=f'labID_{label}')
            
            
if __name__ == '__main__':
    train_all()
    train_groups()
    train_labID_exp()

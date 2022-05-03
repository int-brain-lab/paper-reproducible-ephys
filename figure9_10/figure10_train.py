from tqdm import notebook
import numpy as np
from reproducible_ephys_functions import save_data_path
from figure9_10.utils import static_idx, cov_idx_dict, sim_cov_idx_dict, grouped_cov_idx_dict
from figure9_10.mtnn import initialize_mtnn, run_train


def train(do_train=True, sim_do_train=True):
    data_load_path = save_data_path(figure='figure9_10').joinpath('mtnn_data')
    sim_data_load_path = save_data_path(figure='figure9_10').joinpath('simulated_data')

    feature = np.load(data_load_path.joinpath('train/feature.npy'))

    neuron_order = feature[:, 0, 0]
    feature = feature[:, :, 1:]

    neurons = np.unique(neuron_order)
    n_neurons = neurons.shape[0]
    print('number of neurons: {}'.format(n_neurons))

    INPUT_SIZE_DYNAMIC = feature.shape[-1]-static_idx.shape[0]
    INPUT_SIZE_STATIC = static_idx.shape[0]
    print(INPUT_SIZE_STATIC, INPUT_SIZE_DYNAMIC)

    HIDDEN_SIZE_STATIC = 64
    HIDDEN_SIZE_DYNAMIC = 64
    n_layers = 2
    n_epochs = 100

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


if __name__ == '__main__':
    train()
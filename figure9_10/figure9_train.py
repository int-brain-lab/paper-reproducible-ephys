from figure9_10.utils import static_idx, leave_out_covs_for_glm
from figure9_10.mtnn import initialize_mtnn, run_train
import numpy as np
from reproducible_ephys_functions import save_data_path


def train(do_train=True, do_train_on_glm_covariates=True):

    data_load_path = save_data_path(figure='figure9_10').joinpath('mtnn_data')
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
    remove_cov = None
    only_keep_cov = None

    model = initialize_mtnn(n_neurons=n_neurons,
                            input_size_static=INPUT_SIZE_STATIC,
                            input_size_dynamic=INPUT_SIZE_DYNAMIC,
                            static_bias=True, dynamic_bias=True,
                            hidden_dim_static=HIDDEN_SIZE_STATIC,
                            hidden_dim_dynamic=HIDDEN_SIZE_DYNAMIC, n_layers=n_layers,
                            dropout=0.2)

    if do_train:
        best_epoch, loss_list, val_loss_list = run_train(model,
                                                         data_load_path.joinpath('train/feature.npy'),
                                                         data_load_path.joinpath('train/output.npy'),
                                                         data_load_path.joinpath('validation/feature.npy'),
                                                         data_load_path.joinpath('validation/output.npy'),
                                                         batch_size=512, n_epochs=n_epochs, lr=0.1,
                                                         weight_decay=1e-5,
                                                         remove_cov=remove_cov,
                                                         only_keep_cov=only_keep_cov)

    if do_train_on_glm_covariates:
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
                                                         remove_cov=leave_out_covs_for_glm,
                                                         only_keep_cov=only_keep_cov)


import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from tqdm import notebook, tqdm

from fig_mtnn.utils import static_bool, cov_idx_dict, sim_static_bool, sim_cov_idx_dict
from reproducible_ephys_functions import save_data_path

import random

# torch.manual_seed(7358)
torch.manual_seed(73587)
# np.random.seed(981414)
# random.seed(55777)
np.random.seed(55566)
random.seed(515)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
# g.manual_seed(993342310)
g.manual_seed(99334231)

class MTNN(nn.Module):
    def __init__(self, n_neurons, 
                 input_size_static, input_size_dynamic, 
                 output_size, hidden_dim_static, hidden_dim_dynamic, device,
                 n_layers=1, static_bias=True, dynamic_bias=True, dropout=0.2):
        super(MTNN, self).__init__()

        self.n_neurons = n_neurons
        
        # Defining some parameters
        self.hidden_dim_static = hidden_dim_static
        self.hidden_dim_dynamic = hidden_dim_dynamic
        self.n_layers = n_layers
        self.output_size = output_size
        
        self.fc_static1 = nn.Linear(input_size_static, hidden_dim_static, bias=static_bias)
        self.fc_static2 = nn.Linear(hidden_dim_static, hidden_dim_static, bias=static_bias)
        self.fc_dynamic = nn.Linear(input_size_dynamic, hidden_dim_dynamic, bias=dynamic_bias)

        self.Dropout = torch.nn.Dropout(p=dropout)

        # RNN Layer
        self.rnn = nn.RNN(hidden_dim_dynamic, hidden_dim_dynamic, 
                          n_layers, batch_first=True, 
                          nonlinearity='relu', bidirectional=True,
                          dropout=dropout,
                          bias=dynamic_bias)
        
        # Final fully connected layers
        # fc_list = []
        # for i in range(n_neurons):
        #     fc_list.append(nn.Linear(hidden_dim_dynamic*2+hidden_dim_static, output_size))
        # self.fc_list = nn.ModuleList(fc_list)

        # Final fully connected layers
        fc_list = []
        fc_bias_list = []
        for i in range(n_neurons):
            linear = nn.Linear(hidden_dim_dynamic*2+hidden_dim_static, output_size)
            fc_list.append(linear.weight.T.unsqueeze(0))
            fc_bias_list.append(linear.bias.unsqueeze(0).unsqueeze(0))
        fc_list = torch.cat(fc_list, dim=0)
        fc_bias_list = torch.cat(fc_bias_list, dim=0)
        # print(fc_list.shape)
        
        # neuron_weights = -1 + torch.randn(n_neurons, hidden_dim_dynamic*2+hidden_dim_static, 
        #                                   output_size, requires_grad=True) * 2
        # neuron_weights *= torch.sqrt(torch.Tensor([1/(hidden_dim_dynamic*2+hidden_dim_static)]))

        # neuron_bias = -1 + torch.randn(n_neurons, 1, output_size, requires_grad=True) * 2
        
        self.neuron_weights = torch.nn.Parameter(fc_list)#neuron_weights)
        self.neuron_bias = torch.nn.Parameter(fc_bias_list)#neuron_bias)

        self.ReLU = torch.nn.ReLU()
        
        self.device = device
    
    def forward(self, x_static, x_dynamic, neuron_order):
        
        batch_size = x_static.size(0)
        
        hidden = self.init_hidden(batch_size)
        hidden = hidden.to(self.device)
        
        #print(x_static.shape)
        
        out_static = self.fc_static1(x_static)
        out_static = self.ReLU(out_static)
        out_static = self.Dropout(out_static)
        
        out_static = self.fc_static2(out_static)
        out_static = self.ReLU(out_static)
        out_static = self.Dropout(out_static)
        #print(out_static.shape)
        #print(x_dynamic.shape)

        out_dynamic = self.fc_dynamic(x_dynamic)
        out_dynamic = self.ReLU(out_dynamic)
        out_dynamic = self.Dropout(out_dynamic)
        #print(out_dynamic.shape)

        out_dynamic, _ = self.rnn(out_dynamic, hidden)
        #print(out_dynamic.shape)

        out_dynamic = self.ReLU(out_dynamic)
        out_dynamic = self.Dropout(out_dynamic)

        # output = []
        # for i, neuron_id in enumerate(neuron_order):
        #     neuron_i = int(neuron_id.item())
        #     out_static_i = out_static[i].unsqueeze(0)
        #     out_dynamic_i = out_dynamic[i]
        #     out_static_i = torch.tile(out_static_i, (out_dynamic_i.shape[0], 1))
        #     final_input = torch.cat((out_static_i, out_dynamic_i), dim=1)
        #     y = self.fc_list[neuron_i](final_input).T
        #     print(y.shape)
        #     output.append(y)

        # output = torch.cat(output)
        # print(output.shape)

        # output = torch.zeros((len(neuron_order), out_dynamic.shape[1]))
        # output = output.to(self.device)
        # for neuron_id in range(self.n_neurons):
        #     idx = neuron_order == neuron_id
        #     if idx.sum() == 0:
        #         continue
        #     out_static_i = out_static[idx].unsqueeze(1)
        #     out_dynamic_i = out_dynamic[idx]
        #     out_static_i = torch.tile(out_static_i, (1, out_dynamic_i.shape[1], 1))
        #     final_input = torch.cat((out_static_i, out_dynamic_i), dim=2)
        #     final_layer_output = self.fc_list[neuron_id](final_input)
        #     final_layer_output = final_layer_output.squeeze() #torch.transpose(final_layer_output, 1, 2)
        #     output[idx] = final_layer_output

        neuron_order = neuron_order.to(self.device)
        neuron_order = neuron_order.int()
        # print(neuron_order)
        out_static = torch.tile(out_static.unsqueeze(1), (1, out_dynamic.shape[1], 1))
        final_input = torch.cat((out_static, out_dynamic), dim=2)
        neuronWeights = torch.index_select(self.neuron_weights, 0, neuron_order)
        output = torch.einsum(
            "btd,bdk->btk", final_input, neuronWeights
        ) + torch.index_select(self.neuron_bias, 0, neuron_order)

        output = output.squeeze()
        out = self.ReLU(output)
        #print(out.shape)
        #out = torch.swapaxes(out,0,1)
        
        return out
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward/backward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = Variable(torch.zeros(self.n_layers*2, batch_size, self.hidden_dim_dynamic))
        return hidden


class MTNNDataset(Dataset):
    def __init__(self, neuron_order, feature, output, static_bool):
        self.neuron_order = neuron_order
        self.static_feature = feature[:,0,static_bool]
        self.dynamic_feature = feature[:,:,~static_bool]
        self.output = output
        self.static_bool = static_bool

    def __len__(self):
        return len(self.neuron_order)

    def __getitem__(self, idx):
        neu = self.neuron_order[idx]
        static_f = self.static_feature[idx]
        dynamic_f = self.dynamic_feature[idx]
        target = self.output[idx]
        
        return neu, static_f, dynamic_f, target


def get_device():
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")
    return device


def initialize_mtnn(n_neurons, input_size_static, input_size_dynamic,
                    static_bias, dynamic_bias, hidden_dim_static, 
                    hidden_dim_dynamic, n_layers, dropout=0.2, model_state_dict=None):

    device = get_device()
    
    model = MTNN(n_neurons=n_neurons, input_size_static=input_size_static, 
                 input_size_dynamic=input_size_dynamic,
                 output_size=1, static_bias=static_bias, dynamic_bias=dynamic_bias, 
                 hidden_dim_static=hidden_dim_static, 
                 hidden_dim_dynamic=hidden_dim_dynamic, device=device,
                 n_layers=n_layers, dropout=dropout)
    model.to(device)
    
    if model_state_dict is not None:
        model.load_state_dict(torch.load(model_state_dict))
    
    return model


def leave_out(feature, covs, simulated, return_mask=False):
    cov_idx_map = cov_idx_dict if not simulated else sim_cov_idx_dict
    static = static_bool if not simulated else sim_static_bool
    
    mask = np.ones(feature.shape[-1]).astype(bool)
    for cov in covs:
        i,j = cov_idx_map[cov]
        feature[:,:,i:j] = 0
        mask[i:j] = False
        
    if return_mask:
        return feature, mask[1:][~static]
    else:
        return feature


def only_keep(feature, cov, simulated, return_mask=False):
    cov_idx_map = cov_idx_dict if not simulated else sim_cov_idx_dict
    static = static_bool if not simulated else sim_static_bool
    
    mask = np.ones(feature.shape[-1]).astype(bool)
    i,j = cov_idx_map[cov]
    for idx in range(1, feature.shape[-1]):
        if idx >= i and idx < j:
            continue
        feature[:,:,idx] = 0
        mask[idx] = False
        
    if return_mask:
        return feature, mask[1:][~static]
    else:
        return feature


def shift3D(tensor, shifts: torch.LongTensor):
    # assumes 3D tensor
    n_batches, n_rows, n_cols = tensor.shape

    arange1 = torch.arange(n_rows).view((1,n_rows,1)).repeat((n_batches,1,n_cols))
    arange2 = (arange1 - shifts) % n_rows
    return torch.gather(tensor, 1, arange2)


def shift2D(mat, shifts: torch.LongTensor):
    # assumes 2D matrix
    n_batches, n_rows = mat.shape

    arange1 = torch.arange(n_rows).view((1,n_rows)).repeat((n_batches,1))
    arange2 = (arange1 - shifts) % n_rows
    arange2 = arange2.cuda()
    shifted = torch.gather(mat, 1, arange2)
    
    del arange2
    torch.cuda.empty_cache()
    
    return shifted


def random_pad_shift(dynamic_f, mask):
    
    pre_padding, post_padding = np.random.randint(dynamic_f.shape[1]+1,
                                                  int(dynamic_f.shape[1]*1.5),
                                                  size=2)
    # TODO: use torch.nn.functional.pad
    pre = torch.randn((dynamic_f.shape[0], pre_padding, dynamic_f.shape[-1]))
    post = torch.randn((dynamic_f.shape[0], post_padding, dynamic_f.shape[-1]))
    pre[:,:,~mask] = 0
    post[:,:,~mask] = 0
    padded = torch.cat((pre, dynamic_f, post), dim=1)
    #padded = pad(dynamic_f, (0,0,pre_padding,post_padding))
    
    # random shifting
    shifts = torch.randint(-(pre_padding-dynamic_f.shape[1]),
                           (post_padding-dynamic_f.shape[1]),
                           size=[padded.shape[0],1,1])
    rolled = shift3D(padded, shifts)
    
    return rolled, pre_padding, post_padding, shifts


def add_weight_decay(net, l2_value1, l2_value2):
    decay1, decay2 = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad: continue # frozen weights           
        if 'rnn' in name and 'bias' in name: # rnn bias
            decay2.append(param)
        else: 
            decay1.append(param)
    return [{'params': decay1, 'weight_decay': l2_value1}, {'params': decay2, 'weight_decay': l2_value2}]


def run_train(model, train_feature_path, train_output_path,
              val_feature_path, val_output_path, n_epochs=150, 
              lr=0.1, lr_decay=0.95, momentum=0.9, batch_size=512, 
              clip=2e0, weight_decay=1e-5, valid_loss_min = np.inf,
              model_name_suffix=None, remove_cov=None, 
              only_keep_cov=None, eval_train=False, simulated=False):

    save_remove_cov = map_leave_out_covs_for_glm(remove_cov)

    if model_name_suffix is None:
        model_name = f'state_dict_rem={save_remove_cov}_keep={only_keep_cov}'
    else:
        model_name = f'state_dict_rem={save_remove_cov}_keep={only_keep_cov}_{model_name_suffix}'
    model_name = model_name + '_simulated.pt' if simulated else model_name + '.pt'
    save_path = save_data_path(figure='fig_mtnn').joinpath('trained_models')
    save_path.mkdir(exist_ok=True, parents=True)

    static = static_bool if not simulated else sim_static_bool

    criterion = nn.PoissonNLLLoss(log_input=False)

    #params = add_weight_decay(model, weight_decay, bias_weight_decay)
    #optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_lambda = lambda epoch: lr_decay ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    train_feature = np.load(train_feature_path)
    train_output = np.load(train_output_path)
    val_feature = np.load(val_feature_path)
    val_output = np.load(val_output_path)

    if remove_cov is not None:
        train_feature, mask = leave_out(train_feature, remove_cov, simulated, return_mask=True)
        val_feature = leave_out(val_feature, remove_cov, simulated)
    elif only_keep_cov is not None:
        train_feature, mask = only_keep(train_feature, only_keep_cov, simulated, return_mask=True)
        val_feature = only_keep(val_feature, only_keep_cov, simulated)
    else:
        mask = np.ones(train_feature.shape[-1]-static.sum()-1).astype(bool)

    train_neuron_order = train_feature[:,0,0]
    val_neuron_order = val_feature[:,0,0]
    train_feature = train_feature[:,:,1:]
    val_feature = val_feature[:,:,1:]

    train_data = MTNNDataset(train_neuron_order, train_feature, train_output, static)
    val_data = MTNNDataset(val_neuron_order, val_feature, val_output, static)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, 
                                  shuffle=True, worker_init_fn=seed_worker,
                                  generator=g)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    best_epoch = 0

    loss_list = []
    val_loss_list = []
    model.train()
    for epoch in tqdm(range(1, n_epochs + 1)):
        batch_losses = []
        for neu, static_f, dynamic_f, target in train_dataloader:
            optimizer.zero_grad() # Clears existing gradients from previous epoch
            static_f_gpu = static_f.cuda().float()

            dynamic_f, pre_pads, post_pads, shifts = random_pad_shift(dynamic_f, mask)
            dynamic_f_gpu = dynamic_f.cuda().float()
            target_gpu = target.cuda().float()

            output = shift2D(model(static_f_gpu, dynamic_f_gpu, neu), -shifts[:,:,0])[:,pre_pads:-post_pads]
            loss = criterion(output.reshape(-1), target_gpu.reshape(-1).float())

            loss.backward() # Does backpropagation and calculates gradients
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step() # Updates the weights accordingly
            batch_losses.append(loss.item())

            del static_f_gpu
            del dynamic_f_gpu
            del target_gpu
            del loss
            del output
            torch.cuda.empty_cache()
        scheduler.step()

        if epoch%3 == 1 or epoch == n_epochs:
            print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')

            model.eval()
            with torch.no_grad():

                if eval_train:
                    batch_losses = []
                    for neu, static_f, dynamic_f, target in train_dataloader:
                        static_f_gpu = static_f.cuda().float()
                        dynamic_f_gpu = dynamic_f.cuda().float()
                        target_gpu = target.cuda().float()
                        output = model(static_f_gpu, dynamic_f_gpu, neu)
                        loss = criterion(output.reshape(-1), target_gpu.reshape(-1).float())
                        batch_losses.append(loss.item())

                        del static_f_gpu
                        del dynamic_f_gpu
                        del target_gpu
                        del loss
                        del output
                        torch.cuda.empty_cache()
                    loss_list.append(np.mean(batch_losses))
                    print("Training Loss: {:.4f}".format(loss_list[-1]), end=' ')

                batch_losses=[]
                for neu, static_f, dynamic_f, target in val_dataloader:
                    static_f_gpu = static_f.cuda().float()
                    dynamic_f_gpu = dynamic_f.cuda().float()
                    target_gpu = target.cuda().float()
                    output = model(static_f_gpu, dynamic_f_gpu, neu)
                    loss = criterion(output.reshape(-1), target_gpu.reshape(-1).float())
                    batch_losses.append(loss.item())

                    del static_f_gpu
                    del dynamic_f_gpu
                    del target_gpu
                    del loss
                    del output
                    torch.cuda.empty_cache()
                val_loss_list.append(np.mean(batch_losses))
            model.train()
            print("Validation Loss: {:.4f}".format(val_loss_list[-1]))
            if val_loss_list[-1] <= valid_loss_min:
                torch.save(model.state_dict(), save_path.joinpath(model_name))
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,
                                                                                                val_loss_list[-1]))
                valid_loss_min = val_loss_list[-1]
                best_epoch = epoch

    return best_epoch, loss_list, val_loss_list


def run_eval(model, test_feature_path, test_output_path, test_feature=None,
             batch_size=256, remove_cov=None, only_keep_cov=None, simulated=False):
    
    criterion = nn.PoissonNLLLoss(log_input=False)
    
    if test_feature is None:
        test_feature = np.load(test_feature_path)
    test_output = np.load(test_output_path)
    static = static_bool if not simulated else sim_static_bool
    
    if remove_cov is not None:
        test_feature = leave_out(test_feature, remove_cov, simulated)
    elif only_keep_cov is not None:
        test_feature = only_keep(test_feature, only_keep_cov, simulated)
    
    test_neuron_order = test_feature[:,0,0]
    test_feature = test_feature[:,:,1:]
    
    test_data = MTNNDataset(test_neuron_order, test_feature, test_output, static)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    preds = []
    
    model.eval()
    with torch.no_grad():
        batch_losses = []
        for neu, static_f, dynamic_f, target in test_dataloader:
            static_f_gpu = static_f.cuda().float()
            dynamic_f_gpu = dynamic_f.cuda().float()
            target_gpu = target.cuda().float()
            output = model(static_f_gpu, dynamic_f_gpu, neu)
            preds.append(output.detach().cpu().numpy())
            loss = criterion(output.reshape(-1), target_gpu.reshape(-1).float())
            batch_losses.append(loss.item())

            del static_f_gpu
            del dynamic_f_gpu
            del target_gpu
            torch.cuda.empty_cache()
        loss = np.mean(batch_losses)
    
    return np.concatenate(preds), loss


def compute_score(obs, pred, metric='r2', use_psth=False):
    
    n_neurons = obs.shape[0]
    score = {}
    if metric == 'r2':
        if use_psth:
            score['r2'] = r2_score(obs.mean(1).T, pred.mean(1).T, 
                                   multioutput='raw_values')
        else:
            score['r2'] = r2_score(obs.reshape(n_neurons,-1).T, 
                                   pred.reshape(n_neurons,-1).T,
                                   multioutput='raw_values')
    elif metric == 'pearsonr':
        if use_psth:
            score['pearsonr'] = pearsonr(obs.mean(1), pred.mean(1))[0]
        else:
            score['pearsonr'] = pearsonr(obs.reshape(n_neurons,-1), pred.reshape(n_neurons,-1))[0]
    else:
        raise ValueError(
                f"{metric} not supported!"
            )
    
    return score

def map_leave_out_covs_for_glm(remove_cov):
    # A complete hack because for leave_one_out_glm_cov the filename is too long on windows
    if remove_cov and len(remove_cov) > 10:
        save_remove_cov = ['covs_glm']
    else:
        save_remove_cov = remove_cov
    return save_remove_cov


def load_test_model(model_config, remove_cov, only_keep_cov, 
                    obs_list, preds_shape, metric='r2', 
                    use_psth=False, data_dir='test', 
                    model_name_suffix=None, simulated=False):
    model = initialize_mtnn(n_neurons=model_config['n_neurons'], 
                            input_size_static=model_config['input_size_static'], 
                            input_size_dynamic=model_config['input_size_dynamic'],
                            static_bias=model_config['static_bias'],
                            dynamic_bias=model_config['dynamic_bias'],
                            hidden_dim_static=model_config['hidden_size_static'], 
                            hidden_dim_dynamic=model_config['hidden_size_dynamic'], 
                            n_layers=model_config['n_layers'])
    
    data_load_path = save_data_path(figure='fig_mtnn')
    sim_data_load_path = save_data_path(figure='fig_mtnn')
    model_load_path = save_data_path(figure='fig_mtnn')
    
    if not simulated:
        feature_fname = data_load_path.joinpath(f'mtnn_data/{data_dir}/feature.npy')
        output_fname = data_load_path.joinpath(f'mtnn_data/{data_dir}/output.npy')
    else:
        feature_fname = sim_data_load_path.joinpath(f'simulated_data/{data_dir}/feature.npy')
        output_fname = sim_data_load_path.joinpath(f'simulated_data/{data_dir}/output.npy')

    save_remove_cov = map_leave_out_covs_for_glm(remove_cov)

    if model_name_suffix is None:
        model_name = f'trained_models/state_dict_rem={save_remove_cov}_keep={only_keep_cov}'
    else:
        model_name = f'trained_models/state_dict_rem={save_remove_cov}_keep={only_keep_cov}_{model_name_suffix}'
    model_name = model_name + '_simulated.pt' if simulated else model_name + '.pt'
    model_name = model_load_path.joinpath(model_name)

    model.load_state_dict(torch.load(model_name))
    preds, loss = run_eval(model,feature_fname, output_fname,
                           remove_cov=remove_cov, only_keep_cov=only_keep_cov,
                           simulated=simulated)
    print(f'rem={remove_cov} keep={only_keep_cov} {data_dir} loss: {loss}')
    pred_list = []
    idx = 0
    for sh in preds_shape:
        n = sh[0]*sh[1]
        pred_list.append(preds[idx:idx+n].reshape(sh[:-1]))
        idx += n

    scores = []
    for i in range(len(obs_list)):
        scr = compute_score(obs_list[i],
                          pred_list[i],
                          metric=metric,
                          use_psth=use_psth)[metric]
        scores.extend(list(scr))
    return np.asarray(scores)
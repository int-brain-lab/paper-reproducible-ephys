import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

from tqdm import notebook
from collections import defaultdict

from utils import *

class MTNN(nn.Module):
    def __init__(self, n_neurons, 
                 input_size_static, input_size_dynamic, 
                 output_size, hidden_dim_static, hidden_dim_dynamic, device,
                 n_layers=1, static_bias=True, dynamic_bias=True, dropout=0.1):
        super(MTNN, self).__init__()

        self.n_neurons = n_neurons
        
        # Defining some parameters
        self.hidden_dim_static = hidden_dim_static
        self.hidden_dim_dynamic = hidden_dim_dynamic
        self.n_layers = n_layers
        
        self.fc_static = nn.Linear(input_size_static, hidden_dim_static, bias=static_bias)
        self.fc_dynamic = nn.Linear(input_size_dynamic, hidden_dim_dynamic, bias=dynamic_bias)

        self.Dropout = torch.nn.Dropout(p=dropout)

        # RNN Layer
        self.rnn = nn.RNN(hidden_dim_dynamic, hidden_dim_dynamic, 
                          n_layers, batch_first=True, 
                          nonlinearity='relu', bidirectional=True, bias=dynamic_bias)
        
        #Fully connected layer
        fc_list = []
        for i in range(n_neurons):
            fc_list.append(nn.Linear(hidden_dim_dynamic*2+hidden_dim_static, output_size))
        self.fc_list = nn.ModuleList(fc_list)

        self.ReLU = torch.nn.ReLU()
        
        self.device = device
    
    def forward(self, x_static, x_dynamic, neuron_order):
        
        batch_size = x_static.size(0)
        
        hidden = self.init_hidden(batch_size)
        hidden = hidden.to(self.device)
        
        #print(x_static.shape)
        
        # Passing in the input and hidden state into the model and obtaining outputs
        out_static = self.fc_static(x_static)
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

        output = []

        for i, neuron_id in enumerate(neuron_order):
            neuron_i = int(neuron_id.item())
            out_static_i = out_static[i].unsqueeze(0)
            out_dynamic_i = out_dynamic[i]
            out_static_i = torch.tile(out_static_i, (out_dynamic_i.shape[0], 1))
            final_input = torch.cat((out_static_i, out_dynamic_i), dim=1)
            output.append(self.fc_list[neuron_i](final_input).T)
         
        output = torch.cat(output)
        out = self.ReLU(output)
        #print(out.shape)
        #out = torch.swapaxes(out,0,1)
        
        return out
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
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

def initialize_mtnn(n_neurons, input_size_static, input_size_dynamic,
                    output_size, hidden_dim_static, hidden_dim_dynamic, n_layers):
    # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")
    
    model = MTNN(n_neurons=n_neurons, input_size_static=input_size_static, 
                 input_size_dynamic=input_size_dynamic,
                 output_size=output_size, hidden_dim_static=hidden_dim_static, 
                 hidden_dim_dynamic=hidden_dim_dynamic, device=device,
                 n_layers=n_layers)
    model.to(device)
    
    return model

def leave_out(feature, covs):
    for cov in covs:
        i,j = cov_idx_dict[cov]
        feature[:,:,i:j] = 0
    return feature

def only_keep(feature, cov):
    i,j = cov_idx_dict[cov]
    for idx in range(1, feature.shape[-1]):
        if idx >= i and idx < j:
            continue
        feature[:,:,idx] = 0
    return feature

def random_padding(dynamic_f):
    
    pre_padding, post_padding = np.random.randint(dynamic_f.shape[1],
                                                  dynamic_f.shape[1]*2,
                                                  size=2)
    pre = torch.zeros((dynamic_f.shape[0], pre_padding, dynamic_f.shape[-1]))
    post = torch.zeros((dynamic_f.shape[0], post_padding, dynamic_f.shape[-1]))
    padded = torch.cat((pre, dynamic_f, post), dim=1)
    
    return padded, pre_padding, post_padding

def run_train(model, train_feature_path, train_output_path,
             val_feature_path, val_output_path, n_epochs=150, 
             lr=0.01, batch_size=64, clip=3, remove_cov=None, only_keep_cov=None):
    
    criterion = nn.PoissonNLLLoss(log_input=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    train_feature = np.load(train_feature_path)
    train_output = np.load(train_output_path)
    val_feature = np.load(val_feature_path)
    val_output = np.load(val_output_path)
    
    if remove_cov is not None:
        train_feature = leave_out(train_feature, remove_cov)
        val_feature = leave_out(val_feature, remove_cov)
    elif only_keep_cov is not None:
        train_feature = only_keep(train_feature, only_keep_cov)
        val_feature = only_keep(val_feature, only_keep_cov)
    
    train_neuron_order = train_feature[:,0,0]
    val_neuron_order = val_feature[:,0,0]
    train_feature = train_feature[:,:,1:]
    val_feature = val_feature[:,:,1:]
    
    train_data = MTNNDataset(train_neuron_order, train_feature, train_output, static_bool)
    val_data = MTNNDataset(val_neuron_order, val_feature, val_output, static_bool)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    best_epoch = 0
    
    loss_list = []
    val_loss_list = []
    valid_loss_min = np.Inf
    model.train()
    for epoch in notebook.tqdm(range(1, n_epochs + 1)):
        batch_losses = []
        for neu, static_f, dynamic_f, target in notebook.tqdm(train_dataloader):
            optimizer.zero_grad() # Clears existing gradients from previous epoch
            static_f_gpu = static_f.cuda().float()
            
            dynamic_f, pre_padding, post_padding = random_padding(dynamic_f)
            dynamic_f_gpu = dynamic_f.cuda().float()
            target_gpu = target.cuda().float()
            
            output = model(static_f_gpu, dynamic_f_gpu, neu)[:,pre_padding:-post_padding]
            loss = criterion(output.reshape(-1), target_gpu.reshape(-1).float())
            loss.backward() # Does backpropagation and calculates gradients
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step() # Updates the weights accordingly
            batch_losses.append(loss.item())
            
            del static_f_gpu
            del dynamic_f_gpu
            del target_gpu
            torch.cuda.empty_cache()

        if epoch%5 == 1 or epoch == n_epochs:
            print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')

            model.eval()
            with torch.no_grad():
                
                batch_losses = []
                for neu, static_f, dynamic_f, target in train_dataloader:
                    optimizer.zero_grad() # Clears existing gradients from previous epoch
                    static_f_gpu = static_f.cuda().float()
                    dynamic_f_gpu = dynamic_f.cuda().float()
                    target_gpu = target.cuda().float()
                    output = model(static_f_gpu, dynamic_f_gpu, neu)
                    loss = criterion(output.reshape(-1), target_gpu.reshape(-1).float())
                    batch_losses.append(loss.item())

                    del static_f_gpu
                    del dynamic_f_gpu
                    del target_gpu
                    torch.cuda.empty_cache()
                loss_list.append(np.mean(batch_losses))
                
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
                    torch.cuda.empty_cache()
                val_loss_list.append(np.mean(batch_losses))
            model.train()
            print("Training Loss: {:.4f}".format(loss_list[-1]), end=' ')
            print("Validation Loss: {:.4f}".format(val_loss_list[-1]))
            if val_loss_list[-1] <= valid_loss_min:
                torch.save(model.state_dict(), f'trained_models/state_dict_rem={remove_cov}_keep={only_keep_cov}.pt')
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,
                                                                                                val_loss_list[-1]))
                valid_loss_min = val_loss_list[-1]
                best_epoch = epoch
                
    return best_epoch, loss_list, val_loss_list

def run_eval(model, test_feature_path, test_output_path, 
             batch_size=256, remove_cov=None, only_keep_cov=None):
    
    criterion = nn.PoissonNLLLoss(log_input=False)
    
    test_feature = np.load(test_feature_path)
    test_output = np.load(test_output_path)
    
    if remove_cov is not None:
        test_feature = leave_out(test_feature, remove_cov)
    elif only_keep_cov is not None:
        test_feature = only_keep(test_feature, only_keep_cov)
    
    test_neuron_order = test_feature[:,0,0]
    test_feature = test_feature[:,:,1:]
    
    test_data = MTNNDataset(test_neuron_order, test_feature, test_output, static_bool)
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
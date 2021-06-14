import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

from tqdm import notebook
from collections import defaultdict

from get_features import *

class Model(nn.Module):
    def __init__(self, n_clusters, input_size, output_size, hidden_dim, n_layers, total_length, device):
        super(Model, self).__init__()

        self.n_clusters = n_clusters
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.fc = nn.Linear(input_size, hidden_dim)

        #self.Dropout = torch.nn.Dropout(p=0.3)
        self.bn1 = torch.nn.BatchNorm1d(num_features=hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(num_features=hidden_dim*2)
        
        # Bidirectional RNN Layer
        self.rnn = nn.RNN(hidden_dim, hidden_dim, n_layers, batch_first=True, nonlinearity='relu', bidirectional=True)
        
        #Fully connected layer
        fc_list = []
        for i in range(n_clusters):
            fc_list.append(nn.Linear(hidden_dim*2, output_size))
        self.fc_list = nn.ModuleList(fc_list)

        self.ReLU = torch.nn.ReLU()
        
        self.total_length = total_length
        
        self.device = device
    
    def forward(self, x, seq_len):
        
        batch_order = x[:,0,0]
        x = x[:,:,1:]
        batch_size = x.size(0)
        
        hidden = self.init_hidden(batch_size)
        hidden = hidden.to(self.device)

        # Passing in the input and hidden state into the model and obtaining outputs
        out = self.fc(x)
        
        out = out.permute(0,2,1)
        out = self.bn1(out)
        out = out.permute(0,2,1)
        
        out = self.ReLU(out)
        
        packed = torch.nn.utils.rnn.pack_padded_sequence(out, seq_len, batch_first=True, enforce_sorted=False)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, _ = self.rnn(packed, hidden)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=self.total_length) # unpacked
        
        out = out.permute(0,2,1)
        out = self.bn2(out)
        out = out.permute(0,2,1)
        
        out = self.ReLU(out)
        
        output = None
        for i, batch in enumerate(batch_order):
            batch = int(batch.item())
            if output is None:
                output = self.fc_list[batch](out[i])
            else:
                output = torch.cat((output,self.fc_list[batch](out[i])))
        out = self.ReLU(output)
        
        return out
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = Variable(torch.randn(self.n_layers*2, batch_size, self.hidden_dim))
        return hidden

def define_model(n_clusters, input_size, output_size, hidden_dim, n_layers, total_length):
    # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")
    
    model = Model(n_clusters=n_clusters, input_size=input_size, 
                  output_size=output_size, hidden_dim=hidden_dim, n_layers=n_layers,
                 total_length=total_length, device=device)
    model.to(device)
    
    return model

def do_train(model, train, train_seq_len, train_target,
             val, val_seq_len, val_target, n_epochs=100, lr=0.01):
    
    criterion = nn.PoissonNLLLoss(log_input=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    best_epoch = 0
    
    clip = 3
    batch_size = 64
    loss_list = []
    val_loss_list = []
    valid_loss_min = np.Inf
    model.train()
    for epoch in notebook.tqdm(range(1, n_epochs + 1)):
        batch_loss = []
        permutation = torch.randperm(train.size()[0])
        for i in range(0,train.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            optimizer.zero_grad() # Clears existing gradients from previous epoch
            output = model(train[indices], train_seq_len[indices])
            loss = criterion(output.reshape(-1), train_target[indices].reshape(-1).float())
            loss.backward() # Does backpropagation and calculates gradients
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step() # Updates the weights accordingly
            batch_loss.append(loss.item())
        loss_list.append(np.mean(batch_loss))

        if epoch%3 == 1 or epoch == n_epochs:
            print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
            print("Training Loss: {:.4f}".format(loss_list[-1]), end=' ')

            model.eval()
            with torch.no_grad():
                optimizer.zero_grad() # Clears existing gradients from previous epoch
                output = model(val, val_seq_len)
                loss = criterion(output.reshape(-1), val_target.reshape(-1).float())
                val_loss_list.append(loss.item())
            model.train()
            print("Validation Loss: {:.4f}".format(loss.item()))
            if loss.item() <= valid_loss_min:
                torch.save(model.state_dict(), './state_dict_epoch={}_lr={}.pt'.format(epoch, lr))
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,loss.item()))
                valid_loss_min = loss.item()
                best_epoch = epoch
                
    return best_epoch


def get_effect_sizes(model, test_feature, test_seq_len, covariates, session_idx, cluster_idx, n_samples, seed=0):
    errors = defaultdict(list)
    for i in notebook.tqdm(covariates):

        error = 0
        for j in range(n_samples):
            trial_idx = np.random.randint(0, 50)

            original_feat = test_feature[session_idx, cluster_idx, trial_idx][None]

            test_feat = torch.clone(original_feat)
            
            rem_idx = np.arange(50)
            rem_idx = np.delete(rem_idx, trial_idx)
            
            new_idx = np.random.choice(rem_idx)

            if i == 'lab':
                original_lab = torch.where(test_feat[0,0,lab_offset:xyz_offset] == 1)[0].detach().cpu()

                labs = [0,1,4,6,7]
                new_lab_idx = np.random.choice(labs)

                new_lab = np.zeros(9)
                new_lab[new_lab_idx] = 1
                new_lab = torch.from_numpy(new_lab).cuda().float()

                test_feat[:,:,lab_offset:xyz_offset] = new_lab
            elif i == 'xyz':
                new_xyz = np.random.uniform(0.1, 1, size=3)
                new_xyz = torch.from_numpy(new_xyz).cuda().float()

                test_feat[:,:,xyz_offset:max_ptp_offset] = new_xyz

            elif i == 'reward':
                new_session = np.random.randint(0, 10)
                new_cluster = np.random.randint(0, 10)
                
                new_reward = test_feature[new_session, new_cluster, new_idx][:,reward_offset:wheel_offset]

                test_feat[:,:,reward_offset:wheel_offset] = new_reward

            elif i == 'wheel':
                new_session = np.random.randint(0, 10)
                new_cluster = np.random.randint(0, 10)

                new_wheel = test_feature[new_session, new_cluster, new_idx][:,wheel_offset]

                test_feat[:,:,wheel_offset] = new_wheel

            elif i == 'lick':
                new_session = np.random.randint(0, 10)
                new_cluster = np.random.randint(0, 10)

                new_lick = test_feature[new_session, new_cluster, new_idx][:,lick_offset]

                test_feat[:,:,lick_offset] = new_lick
                
            elif i == 'contrast':
                contrast_val = [0.    , 0.0625, 0.125 , 0.25  , 1.    ]
                new_contrast_val = np.random.choice(contrast_val)
                new_contrast_idx = np.random.choice([0,1])

                new_contrast = np.zeros(2)
                new_contrast[new_contrast_idx] = new_contrast_val
                new_contrast = torch.from_numpy(new_contrast).cuda().float()

                test_feat[:,:,contrast_offset:goCue_offset] = new_contrast
                
            elif i == 'max_ptp':
                new_ptp_val = np.random.uniform(0.1, 1)
                new_ptp = np.zeros(1)
                new_ptp[0] = new_ptp_val
                new_ptp = torch.from_numpy(new_ptp).cuda().float()

                test_feat[:,:,max_ptp_offset] = new_ptp
                
            elif i == 'wf_width':
                new_width_val = np.random.uniform(0.1, 1)
                new_width = np.zeros(1)
                new_width[0] = new_width_val
                new_width = torch.from_numpy(new_width).cuda().float()

                test_feat[:,:,wf_width_offset] = new_width

            torch.manual_seed(seed)
            test_output = model(test_feat, test_seq_len[0][None])

            torch.manual_seed(seed)
            original_output = model(original_feat, test_seq_len[0][None])

            error += torch.mean(torch.abs(original_output - test_output))

            del test_feat
            del test_output
            del original_output
            torch.cuda.empty_cache()

        error_cpu = (error/n_samples).detach().cpu().numpy()

        del error
        torch.cuda.empty_cache()

        errors[i].append(error_cpu)
        
    return errors
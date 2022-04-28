import numpy as np
import pandas as pd
import seaborn as sns
from os.path import join
from collections import defaultdict
import matplotlib.pyplot as plt
import brainbox.io.one as bbone
import sys 
sys.path.append('..')
from reproducible_ephys_functions import query, save_data_path
from tqdm import notebook
import brainbox as bb
from ibllib.io import spikeglx
import brainbox.behavior.wheel as wh
from brainbox import singlecell
from brainbox.metrics.single_units import spike_sorting_metrics
from one.api import ONE
import alf

from ibllib.qc.camera import CameraQC
import os
from scipy.stats import zscore
from collections import Counter

from utils import *

def select_high_fr_neurons(feature, output, clusters, 
                           neuron_id_start=0, threshold1=5.0, threshold2=2.0, max_n_neurons=15):
    #select = output.mean(1).max(1) >= threshold
    select = np.logical_and(output.mean(1).max(1) >= threshold1, np.mean(output, axis=(1,2)) >= threshold2)
    feature_subset = feature[select]
    if feature_subset.shape[0] > max_n_neurons:
        select2 = np.random.choice(np.arange(feature_subset.shape[0]), size=max_n_neurons, replace=False)
    else:
        select2 = np.arange(feature_subset.shape[0])
    feature_subset = feature_subset[select2]
    for i in range(feature_subset.shape[0]):
        feature_subset[i,:,:,0] = neuron_id_start+i
    
    return feature_subset, output[select][select2], clusters[select][select2]

def preprocess_feature(feature_concat):
    # normalize truncated_feature: dlc features + xyz location
    # first normalize xyz
    x_max = feature_concat[:,:,xyz_offset].max()
    x_min = feature_concat[:,:,xyz_offset].min()
    y_max = feature_concat[:,:,xyz_offset+1].max()
    y_min = feature_concat[:,:,xyz_offset+1].min()
    z_max = feature_concat[:,:,xyz_offset+2].max()
    z_min = feature_concat[:,:,xyz_offset+2].min()

    feature_concat[:,:,xyz_offset] = 0.1 + 0.9*(feature_concat[:,:,xyz_offset] - x_min) / (x_max - x_min)
    feature_concat[:,:,xyz_offset+1] = 0.1 + 0.9*(feature_concat[:,:,xyz_offset+1] - y_min) / (y_max - y_min)
    feature_concat[:,:,xyz_offset+2] = 0.1 + 0.9*(feature_concat[:,:,xyz_offset+2] - z_min) / (z_max - z_min)

    # next normalize dlc features
    for i in range(stimulus_offset - paw_offset):
        idx = paw_offset+i

        feature_min = feature_concat[:,:,idx].min()
        feature_max = feature_concat[:,:,idx].max()

        feature_concat[:,:,idx] = 0.1 + 0.9*(feature_concat[:,:,idx] - feature_min) / (feature_max - feature_min)

    # next normalize wheel
    wheel_min = feature_concat[:,:,wheel_offset].min()
    wheel_max = feature_concat[:,:,wheel_offset].max()

    feature_concat[:,:,wheel_offset] = -1 + 2*(feature_concat[:,:,wheel_offset] - wheel_min) / (wheel_max - wheel_min)

    # next normalize max_ptp
    max_ptp_min = feature_concat[:,:,max_ptp_offset].min()
    max_ptp_max = feature_concat[:,:,max_ptp_offset].max()

    feature_concat[:,:,max_ptp_offset] = 0.1 + 0.9*(feature_concat[:,:,max_ptp_offset] - max_ptp_min) / (max_ptp_max - max_ptp_min)

    # next normalize wf_width
    wf_width_min = feature_concat[:,:,wf_width_offset].min()
    wf_width_max = feature_concat[:,:,wf_width_offset].max()

    feature_concat[:,:,wf_width_offset] = 0.1 + 0.9*(feature_concat[:,:,wf_width_offset] - wf_width_min) / (wf_width_max - wf_width_min)
    
    # noise
    noise = np.random.normal(loc=0.0, scale=1.0, size=feature_concat.shape[:-1])
    feature_concat[:,:,noise_offset] = noise
    
    return feature_concat


one = ONE()

mtnn_eids = get_mtnn_eids()
print(list(mtnn_eids.keys()))

feature_list, output_list, cluster_number_list, session_list, trial_number_list = load_original(mtnn_eids)

total_n_neurons = 0
shape_list = []
output_subset_list = []
cluster_subset_list = []
for i in notebook.tqdm(range(len(mtnn_eids))):
    feature_subset, output_subset, clusters_subset = select_high_fr_neurons(feature_list[i], 
                                                                           output_list[i],
                                                                           cluster_number_list[i],
                                                                           neuron_id_start=total_n_neurons,
                                                                           threshold1=8.0,
                                                                           threshold2=2.5,
                                                                           max_n_neurons=15)
    total_n_neurons += feature_subset.shape[0]
    print('{}/{} remaining'.format(feature_subset.shape[0],feature_list[i].shape[0]))
    print('{}/{} removed'.format(feature_list[i].shape[0]-feature_subset.shape[0],feature_list[i].shape[0]))
    shape_list.append(feature_subset.shape)
    output_subset_list.append(output_subset)
    cluster_subset_list.append(clusters_subset)
    
    if i == 0:
        feature_concat = feature_subset.reshape((-1,)+feature_subset.shape[-2:])
    else:
        feature_concat = np.concatenate((feature_concat, feature_subset.reshape((-1,)+feature_subset.shape[-2:])))
print('feature_concat shape: {}'.format(feature_concat.shape))
print(f'number of neurons left: {total_n_neurons}')

preprocessed_feature = preprocess_feature(feature_concat)
print(preprocessed_feature.shape)

preprocessed_feature_list = []
idx = 0
for sh in shape_list:
    n = sh[0]*sh[1]
    preprocessed_feature_list.append(preprocessed_feature[idx:idx+n].reshape(sh))
    idx += n
    
train_shape_list = []
val_shape_list = []
test_shape_list = []

train_bool_list = []
val_bool_list = []
test_bool_list = []

train_trial_list = []
val_trial_list = []
test_trial_list = []

train_feature = []
val_feature = []
test_feature = []

train_output = []
val_output = []
test_output = []

for i in notebook.tqdm(range(len(mtnn_eids))):
    try:
        print(session_list[i]['session']['id'])
    except:
        print(session_list[i].tolist()['session']['id'])
    
    n_trials = preprocessed_feature_list[i].shape[1]
    n_test = int(n_trials*0.2)
    n_train = int((n_trials-n_test)*0.8)
    n_val = n_trials - n_train - n_test
    
    sh = shape_list[i]
    train_shape_list.append((sh[0],n_train,)+sh[-2:])
    val_shape_list.append((sh[0],n_val,)+sh[-2:])
    test_shape_list.append((sh[0],n_test,)+sh[-2:])
    
    test_idx = np.random.choice(np.arange(n_trials), size=n_test, replace=False)
    test_bool = np.zeros(n_trials).astype(bool)
    test_bool[test_idx] = True
    
    train_idx = np.random.choice(np.arange(n_trials)[~test_bool], size=n_train, replace=False)
    train_bool = np.zeros(n_trials).astype(bool)
    train_bool[train_idx] = True
    
    val_bool = np.zeros(n_trials).astype(bool)
    val_bool[~np.logical_or(test_bool, train_bool)] = True
    
    train_bool_list.append(train_bool)
    val_bool_list.append(val_bool)
    test_bool_list.append(test_bool)
    
    train_trial_list.append(trial_number_list[i][train_bool])
    val_trial_list.append(trial_number_list[i][val_bool])
    test_trial_list.append(trial_number_list[i][test_bool])
    
    train_feature.append(preprocessed_feature_list[i][:,train_bool].reshape((-1,)+sh[-2:]))
    val_feature.append(preprocessed_feature_list[i][:,val_bool].reshape((-1,)+sh[-2:]))
    test_feature.append(preprocessed_feature_list[i][:,test_bool].reshape((-1,)+sh[-2:]))
    
    train_output.append(output_subset_list[i][:,train_bool].reshape(-1, sh[-2]))
    val_output.append(output_subset_list[i][:,val_bool].reshape(-1, sh[-2]))
    test_output.append(output_subset_list[i][:,test_bool].reshape(-1, sh[-2]))
    
    
save_path = save_data_path(figure='figure8').joinpath('mtnn_data')
save_path.mkdir(exist_ok=True, parents=True)

save_path_train = save_data_path(figure='figure8').joinpath('mtnn_data/train')
save_path_train.mkdir(exist_ok=True, parents=True)

save_path_val = save_data_path(figure='figure8').joinpath('mtnn_data/validation')
save_path_val.mkdir(exist_ok=True, parents=True)

save_path_test = save_data_path(figure='figure8').joinpath('mtnn_data/test')
save_path_test.mkdir(exist_ok=True, parents=True)


np.save(save_path_train.joinpath('shape.npy'), np.asarray(train_shape_list))
np.save(save_path_val.joinpath('shape.npy'), np.asarray(val_shape_list))
np.save(save_path_test.joinpath('shape.npy'), np.asarray(test_shape_list))

np.save(save_path_train.joinpath('bool.npy'), np.asarray(train_bool_list, dtype=object))
np.save(save_path_val.joinpath('bool.npy'), np.asarray(val_bool_list, dtype=object))
np.save(save_path_test.joinpath('bool.npy'), np.asarray(test_bool_list, dtype=object))

np.save(save_path_train.joinpath('trials.npy'), np.asarray(train_trial_list, dtype=object))
np.save(save_path_val.joinpath('trials.npy'), np.asarray(val_trial_list, dtype=object))
np.save(save_path_test.joinpath('trials.npy'), np.asarray(test_trial_list, dtype=object))

np.save(save_path_train.joinpath('feature.npy'), np.concatenate(train_feature))
np.save(save_path_val.joinpath('feature.npy'), np.concatenate(val_feature))
np.save(save_path_test.joinpath('feature.npy'), np.concatenate(test_feature))

np.save(save_path_train.joinpath('output.npy'), np.concatenate(train_output))
np.save(save_path_val.joinpath('output.npy'), np.concatenate(val_output))
np.save(save_path_test.joinpath('output.npy'), np.concatenate(test_output))

np.save(save_path.joinpath('session_info.npy'), np.asarray(session_list))
np.save(save_path.joinpath('clusters.npy'), np.asarray(cluster_subset_list, dtype=object))
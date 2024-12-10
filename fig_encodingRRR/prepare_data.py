from utils.download_data import load_data_from_pid, filter_trials_func, filter_neurons_func
from reproducible_ephys_functions import filter_recordings

import numpy as np
from tqdm import tqdm
import os, glob, shutil

from one.api import ONE
from iblatlas.atlas import AllenAtlas


cache_folder = "./data/cache/"
data_folder = "./data/rep_ephys_22032024/downloaded/"

one = ONE(base_url='https://openalyx.internationalbrainlab.org',
          password='international', 
          cache_dir=cache_folder)


ba = AllenAtlas()
br = ba.regions

filtering_criteria = {'min_regions': 0, 'min_lab_region': 3, 'min_rec_lab': 0, 'min_neuron_region': 4, 'freeze': 'freeze_2024_03'}
df_filt = filter_recordings(**filtering_criteria)
pids = df_filt[df_filt['permute_include'] == 1].pid.unique()

checked_pids = []
for pid in tqdm(pids):

    neural_fname = f"neural_data_{pid}.npz"
    beh_fname = f"beh_data_{pid}.npz"
    if os.path.isfile(os.path.join(data_folder, neural_fname)) and os.path.isfile(os.path.join(data_folder, beh_fname)):
        checked_pids.append(pid)
        continue

    if pid in checked_pids:
        print("already checked this eid, continue")
        continue
    checked_pids.append(pid)


    data_ret = load_data_from_pid(pid, one, ba,
                                    lambda _: filter_trials_func(_,
                                                                remove_timeextreme_event=[True, 0.8],
                                                                remove_no_choice=True),
                                    lambda _: filter_neurons_func(_,
                                                                remove_frextreme=(True, .1, 50.),
                                                                only_goodneuron=(False),
                                                                only_area=(False)),
                                    min_neurons=5, 
                                    spsdt=0.01, Twindow=1.2, t_bf_stimOn=0.2, # sec
                                    load_motion_energy=True, load_wheel_velocity=True,
                                    load_tongue=True)
    if data_ret is None:
        continue


    np.savez(os.path.join(data_folder, neural_fname),
                spike_count_matrix=data_ret['spike_count_matrix'],
                clusters_g=data_ret['clusters_g'],
                eid=pid,
                )
    

    np.savez(os.path.join(data_folder, beh_fname),
                behavior=data_ret['behavior'],
                timeline=data_ret['timeline'],
                eid=pid,
                wheel_vel=data_ret['wheel_vel'],
                whisker_motion=data_ret['whisker_motion'],
                licks=data_ret["licks"],
                )


    for fname in glob.glob(os.path.join(cache_folder, "*")):
        if os.path.isdir(fname):
            shutil.rmtree(fname)

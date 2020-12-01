import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
from oneibl.one import ONE
import pandas as pd
import alf.io
from colorama import Fore, Style

'''
This script allows to plot RMS (root mean square) of raw 
ephys signals in the AP and LFP band. 
It further has a function to load in eid and probes
from an overview excel sheet.
'''


def get_repeated_site_eids_from_Mayos_excel():

   # Save the excel sheet of the repeated sites locally then read via pandas 
   # https://docs.google.com/spreadsheets/d/1pRLFvyVgmIJfKSX4GqmmAFuNRTpU1NvMlYOijq7iNIA/edit#gid=0 

    one = ONE()
    s=pd.read_excel('/home/mic/Desktop/Repeated_Site_Status.xlsx')
    eids = []
    for i in range(len(s['Subject'].array)):
        ks2 = s['ks2'].array[i]
        hist = s['histology'].array[i]
        #dlc = s['dlc'].array[i]
        if ks2 and hist:
            sub = s['Subject'].array[i]
            dat = s['Date'].array[i]
            eid = one.search(subject = sub, date = dat)[0]
            prob = s['Probe'].array[i]
            eids.append([eid, prob])
            
    return eids


def plot_rms(eid, probe_label):

    # https://int-brain-lab.github.io/iblenv/notebooks_external/docs_get_rms_data.html
    
    #plt.ion() 
    plt.ioff()     

    # instantiate ONE
    one = ONE()

    # Specify the dataset types of interest
    dtypes = ['_iblqc_ephysTimeRms.rms',
              '_iblqc_ephysTimeRms.timestamps',
              'channels.rawInd',
              'channels.localCoordinates']

    # Download the data and get paths to downloaded data
    _ = one.load(eid, dataset_types=dtypes, download_only=True)
    ephys_path = one.path_from_eid(eid).joinpath('raw_ephys_data', probe_label)
    alf_path = one.path_from_eid(eid).joinpath('alf', probe_label)

    session_name = '_'.join(str(ephys_path).split('/')[5:10])
    # Index of good recording channels along probe
    chn_inds = np.load(alf_path.joinpath('channels.rawInd.npy'))
    # Position of each recording channel along probe
    chn_pos = np.load(alf_path.joinpath('channels.localCoordinates.npy'))
    # Get range for y-axis
    depth_range = [np.min(chn_pos[:, 1]), np.max(chn_pos[:, 1])]

    # RMS data associated with AP band of data
    rms_ap = alf.io.load_object(ephys_path, 'ephysTimeRmsAP', namespace='iblqc')
    rms_ap_data = 20* np.log10(rms_ap['rms'][:, chn_inds] * 1e6)  # convert to uV

    # Get levels for colour bar and x-axis
    ap_levels = np.quantile(rms_ap_data, [0.1, 0.9])
    ap_time_range = [rms_ap['timestamps'][0], rms_ap['timestamps'][-1]]

    # RMS data associated with LFP band of data
    rms_lf = alf.io.load_object(ephys_path, 'ephysTimeRmsLF', namespace='iblqc')
    rms_lf_data = rms_lf['rms'][:, chn_inds] * 1e6  # convert to uV

    lf_levels = np.quantile(rms_lf_data, [0.1, 0.9])
    lf_time_range = [rms_lf['timestamps'][0], rms_lf['timestamps'][-1]]

    # Create figure
    fig, ax = plt.subplots(2, 1, figsize=(6, 8))
    # Plot the AP rms data
    ax0 = ax[0]

    rms_ap_plot = ax0.imshow(rms_ap_data.T, extent=np.r_[ap_time_range, depth_range],
                             cmap='plasma', vmin=0, vmax=100, origin='lower')                             
                             
    cbar_ap = fig.colorbar(rms_ap_plot, ax=ax0)
    cbar_ap.set_label('AP RMS (uV)')
    ax0.set_xlabel('Time (s)')
    ax0.set_ylabel('Depth along probe (um)')
    ax0.set_title('RMS of AP band')

    # Plot the LFP rms data
    ax1 = ax[1]
    rms_lf_plot = ax1.imshow(rms_lf_data.T, extent=np.r_[lf_time_range, depth_range],
                             cmap='inferno', vmin=0, vmax=1500, origin='lower')
    cbar_lf = fig.colorbar(rms_lf_plot, ax=ax1)
    cbar_lf.set_label('LFP RMS (uV)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Depth along probe (um)')
    ax1.set_title('RMS of LFP band')
    
    plt.suptitle('%s_%s \n %s' %(eid, probe_label, session_name))
    plt.savefig('/home/mic/saturation_analysis/rms_plots/%s_%s.png' %(eid, probe_label))
    #plt.show()


def plot_all():

    eids = get_repeated_site_eids_from_Mayos_excel()
    probs = []
    for eid in eids:
        try:
            plot_rms(eid[0], eid[1])
        except Exception as e:     
            print(f'{Fore.RED}{Style.BRIGHT}EEEEEERRRRRROOORRRR!!!!!') 
            print(e)
            probs.append([eid,e])
    return probs             


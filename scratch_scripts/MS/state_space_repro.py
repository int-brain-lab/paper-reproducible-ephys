from one.api import ONE
from brainbox.io.one import load_channel_locations,load_spike_sorting_fast
from brainbox.processing import bincount2D
from iblatlas.regions import regions_from_allen_csv, BrainRegions
import iblatlas.atlas as atlas
import gc

import os
import numpy as np
from scipy import stats
from pathlib import Path
from collections import Counter, ChainMap
import math
from scipy.stats import pearsonr, spearmanr, percentileofscore
from copy import deepcopy
import pandas as pd
import random
from datetime import datetime

from sklearn.decomposition import PCA
from scipy.stats import zscore
import itertools

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch
from ipywidgets import interact
from sklearn.metrics import r2_score
import umap
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import string
from itertools import combinations
from statsmodels.stats.multitest import multipletests

T_BIN = 0.02  # time bin size in seconds
one = ONE()


def get_repeated_sites():
    one = ONE()
    STR_QUERY = 'probe_insertion__session__projects__name__icontains,ibl_neuropixel_brainwide_01,' \
                'probe_insertion__session__qc__lt,50,' \
                '~probe_insertion__json__qc,CRITICAL,' \
                'probe_insertion__session__n_trials__gte,400'
    all_sess = one.alyx.rest('trajectories', 'list', provenance='Planned',
                                  x=-2243, y=-2000, theta=15,
                                  django=STR_QUERY)
    #eids = [s['session']['id'] for s in all_sess]
    sess = [[s['session']['id'],s['probe_name']] for s in all_sess]

    r=np.load('/home/mic/repro_manifold/repeated_site_eids.npy')

    s2 = [x for x in sess if x[0] in r]

    return s2


def labs_maps():
    lab_number_map = {'cortexlab': 'Lab 1', 'mainenlab': 'Lab 2',
                      'zadorlab': 'Lab 3',
                      'churchlandlab': 'Lab 4', 'angelakilab': 'Lab 5',
                      'wittenlab': 'Lab 6',
                      'hoferlab': 'Lab 7', 'mrsicflogellab': 'Lab 7',
                      'danlab': 'Lab 8',
                      'steinmetzlab': 'Lab 9', 'churchlandlab_ucla': 'Lab 10'}
    institution_map = {'cortexlab': 'UCL', 'mainenlab': 'CCU',
                        'zadorlab': 'CSHL (Z)',
                       'churchlandlab': 'CSHL (C)', 'angelakilab': 'NYU',
                       'wittenlab': 'Princeton', 'hoferlab': 'SWC',
                       'mrsicflogellab': 'SWC',
                       'danlab': 'Berkeley', 'steinmetzlab': 'UW',
                       'churchlandlab_ucla': 'UCLA'}
    colors = np.concatenate([sns.color_palette("Set1")[0:4], sns.color_palette("Set1")[6:],
                             [[0, 0, 0]], [sns.color_palette("Dark2")[3]],
                             [sns.color_palette("Set2")[0]]])
    institutions = ['UCL', 'CCU', 'CSHL (Z)', 'CSHL (C)', 'NYU', 'Princeton',
                    'SWC', 'Berkeley','UW', 'UCLA']
    institution_colors = {}
    for i, inst in enumerate(institutions):
        institution_colors[inst] = colors[i]
    return lab_number_map, institution_map, institution_colors


def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or
    math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx


def get_acronyms_per_eid(eid, probe=None, new_spike = False):

    T_BIN = 1
    one = ONE()

    As = {}

    if probe is None:
        dsets = one.list_datasets(eid)
        r = [x.split('/') for x in dsets if 'probe' in x]
        rr = [item for sublist in r for item in sublist
              if 'probe' in item and '.' not in item]
        probes = list(Counter(rr))

    else:
        probes = [probe]

    for probe in probes:

        if new_spike:
            spikes, clusters, channels = load_spike_sorting_fast(eid=eid,
                        one=one,  probe=probe, spike_sorter='pykilosort')

            spikes = spikes[probe]
            clusters = clusters[probe]
            channels = channels[probe]
            if spikes == {}:
                print(f'!!! No new spike sorting for {probe} !!!')
                return

        else:
            spikes = one.load_object(eid, 'spikes',
                collection=f'alf/{probe}',attribute=['times','clusters'])
            clusters = one.load_object(eid, 'clusters', collection=f'alf/{probe}',
                attribute=['channels'])


        R, times, Clusters = bincount2D(
            spikes['times'], spikes['clusters'], T_BIN)


        # Get for each cluster the location acronym
        cluster_chans = clusters['channels'][Clusters]
        els = load_channel_locations(eid, one=one)
        ids = els[probe]['atlas_id'][cluster_chans]
        br = atlas.BrainRegions()
        mapped_ids = br.remap(ids, source_map='Allen', target_map='Beryl')
        id_info = br.get(ids=mapped_ids)
        acronyms2 = id_info['acronym']
        acronyms = combine_regions(acronyms2)

        As[probe] = acronyms

    return As


def combine_regions(regions):
    """
    Combine all layers of cortex and the dentate gyrus molecular and granular layer
    Combine VISa and VISam into PPC
    """
    remove = ['1', '2', '3', '4', '5', '6a', '6b', '/']
    for i, region in enumerate(regions):
        if region[:2] == 'CA':
            continue
        if (region == 'DG-mo') or (region == 'DG-sg') or (region == 'DG-po'):
            regions[i] = 'DG'
        for j, char in enumerate(remove):
            regions[i] = regions[i].replace(char, '')
        if (regions[i] == 'VISa') | (regions[i] == 'VISam'):
            regions[i] = 'PPC'
    return regions


def bin_neural(eid, double=True, probe=None, reg=None, query_type='remote',
               new_spike=False):
    '''
    bin neural activity; combine probes or pick single region
    '''
    one = ONE()

    if probe is not None:
        double = False

    if double:
        print(f'combining data from both probes, eid = {eid}')
        sks = []
        dsets = one.list_datasets(eid)
        r = [x.split('/') for x in dsets if 'probe' in x]
        rr = [item for sublist in r for item in sublist
              if 'probe' in item and '.' not in item]

        if len(list(Counter(rr))) != 2:
            print("not two probes present, using one only")


        for probe in list(Counter(rr)):

            if new_spike:
                spikes, _, _ = load_spike_sorting_fast(eid=eid,
                            one=one,  probe=probe, spike_sorter='pykilosort')

                spikes = spikes[probe]

                if spikes == {}:
                    print(f'!!! No new spike sorting for {probe} !!!')
                    return

            else:
                spikes = one.load_object(eid, 'spikes',
                    collection=f'alf/{probe}',attribute=['times','clusters'])

            sks.append(spikes)

        if len(sks) == 1:
            R, times, _ = bincount2D(sks[0]['times'], sks[0]['clusters'], T_BIN)
            D = R.T

        else:
            # add max cluster of p0 to p1, then concat, sort
            max_cl0 = max(sks[0]['clusters'])
            sks[1]['clusters'] = sks[1]['clusters'] + max_cl0 + 1

            times_both = np.concatenate([sks[0]['times'],sks[1]['times']])
            clusters_both = np.concatenate([sks[0]['clusters'],sks[1]['clusters']])

            t_sorted = np.sort(times_both)
            c_ordered = clusters_both[np.argsort(times_both)]

            print('binning data')
            R, times, _ = bincount2D(t_sorted, c_ordered, T_BIN)

            D = R.T

    else:
        print('single probe')
        if new_spike:
            spikes, clusters, channels = load_spike_sorting_fast(eid=eid,
                        one=one,  probe=probe, spike_sorter='pykilosort')

            spikes = spikes[probe]
            clusters = clusters[probe]
            channels = channels[probe]
            if spikes == {}:
                print(f'!!! No new spike sorting for {probe} !!!')
                return

        else:
            spikes = one.load_object(eid, 'spikes',
                collection=f'alf/{probe}',attribute=['times','clusters'])
            clusters = one.load_object(eid, 'clusters', collection=f'alf/{probe}',
                attribute=['channels'])


        # bin spikes
        R, times, Clusters = bincount2D(
            spikes['times'], spikes['clusters'], T_BIN)

        D = R.T

        if reg is not None:
            print(f'region: {reg}')
            # Get for each cluster the location x,y,z
            cluster_chans = clusters['channels'][Clusters]
            els = load_channel_locations(eid, one=one)
            ids = els[probe]['atlas_id'][cluster_chans]
            br = atlas.BrainRegions()
            mapped_ids = br.remap(ids, source_map='Allen', target_map='Beryl')
            id_info = br.get(ids=mapped_ids)
            acronyms = id_info['acronym']
            acronyms2 = combine_regions(acronyms)
            m_ask = acronyms2 == reg
            D = D[:,m_ask]

    return D, times



def PSTH(eid, duration=1.5 ,lag=-0.5, double=True, probe=None,
             reg=None, split='choice', new_spike=True):
    #eid = '56b57c38-2699-4091-90a8-aba35103155e'
    #eid,reg,probe='671c7ea7-6726-4fbe-adeb-f89c2c8e489b','GRN','probe00'

    '''
    combine both probes into binned activity
    then do PSTH
    then keep only bins in trials,
    keep block id info for each bin

    choice_split: instead of blocks, color activity by choice
        and deactivate pseudo sessions

    now aligned to motion onset

    '''

    one = ONE()
    D, times = bin_neural(eid, double=double, probe=probe,
                          reg=reg, new_spike=new_spike)

    obs, n_clus = D.shape
    print(f'session duration: {np.round(obs*T_BIN,0)} sec = '
          f'{np.round(obs*T_BIN/60,2)} min; n_clus = {n_clus}; '
          f'T_BIN = {T_BIN} sec')

    if n_clus < 3:
        raise TypeError("less than 3 neurons in this region")
    # Get lists of trials
    neuleft, neuright = [], []

    print('cutting data')
    wheelMoves = one.load_object(eid, 'wheelMoves')
    A = wheelMoves['intervals'][:,0]
    trials = one.load_object(eid, 'trials')
    evts = ['stimOn_times', 'feedback_times', 'probabilityLeft',
            'choice', 'feedbackType']

    for tr in range(len(trials['intervals'])):

        if np.isnan(trials['contrastLeft'][tr]):
            cont = trials['contrastRight'][tr]
            side = 0  # right side stimulus
        else:
            cont = trials['contrastLeft'][tr]
            side = 1  # left side stimulus

        # skip trial if any key info is nan
        if any(np.isnan([trials[k][tr] for k in evts])):
            continue

        # skip trial if duration is too long
        if trials['feedback_times'][tr] - trials['stimOn_times'][tr] > 10:
            continue

#        # skip left side trials
#        if side == 1:
#            continue

        # skip incorrect trial
        if trials['feedbackType'][tr] == 0:
            continue


        b = trials['stimOn_times'][tr]
        c = trials['feedback_times'][tr]
        # making sure the motion onset time is in a coupled interval
        ind = np.where((A > b) & (A < c), True, False)
        if all(~ind):
            #print(f'non-still start, trial {tr} and eid {eid}')
            continue
        a = A[ind][0]

        start_idx = find_nearest(times,a + lag)#trials['stimOn_times'][tr]
        end_idx = start_idx + int(duration/T_BIN)

        if split == 'RT':

            rt = a - b
            if rt < 0.1:
                neuleft.append(D[start_idx:end_idx])
            if rt > 0.2:
                neuright.append(D[start_idx:end_idx])


        if split == 'choice':
            # split trials

            if trials['choice'][tr] == 1:
                neuleft.append(D[start_idx:end_idx])
            if trials['choice'][tr] == -1:
                neuright.append(D[start_idx:end_idx])
            else:
                continue

        if split == 'block':
            # split by block
            if trials['probabilityLeft'][tr] == 0.5:
                continue

            if trials['probabilityLeft'][tr] == 0.8:
                neuleft.append(D[start_idx:end_idx])
            if trials['probabilityLeft'][tr] == 0.2:
                neuright.append(D[start_idx:end_idx])

    if split == 'choice':
        print('left choice trials: ',len(neuleft),
              '; right choice trials: ', len(neuright))
    if split == 'block':
        print('pleft0.8 trials: ',len(neuleft),
              '; pleft0.2 trials: ', len(neuright))

    print('initial raw trial number ',len(trials['intervals']))
    # as a control, bipartite trials randomly

    neuleft = np.array(neuleft).mean(axis=0)
    neuright = np.array(neuright).mean(axis=0)

    neus = np.concatenate([neuleft,neuright])

    return [neuleft,neuright]


def get_probe_age(pid):

    eid,pname = one.pid2eid(pid)
    probe_insertions = one.load_dataset(eid, 'probes.description')

    serial = [y['serial'] for y in probe_insertions if y['label']==pname][0]

    sess = one.alyx.rest('insertions', 'list', serial=serial)
    d_0 = [x['session_info']['start_time'] for x in sess][-1].split('T')[0]
    then = datetime.strptime(d_0, "%Y-%m-%d")
    now  = datetime.now()
    duration = now - then
    return duration.days


'''################################
batch
######################'''



def single_cell_embedding(duration=1.5, lag=-0.5, new_spike=True,
                          split='RT', norm_=True):

    '''
    PCA on time (choice left/right psth concatenated)
    to show single cell contribution to first PCs
    '''

    errs = []
    sess = get_repeated_sites()
    one = ONE()
    D = {}
    regs = ['PPC', 'CA1', 'DG', 'LP', 'PO', None]
    for reg in regs:

        neus = []  # psth data
        ns = []  # number of clusters
        labs = []

        for ses in sess:
            try:
                w = PSTH(ses[0],duration,lag,probe=ses[1],reg=reg,
                    new_spike=new_spike,split=split)
                neus1 = np.concatenate(w)
                neus.append(neus1)
                ns.append(neus1.shape[1])
                labs.append(str(one.eid2path(ses[0])).split('/')[4])
            except:
                lab = str(one.eid2path(ses[0])).split('/')[4]
                print(ses, lab,
                     '||||||###### went wrong ######||||||')
                errs.append([ses,lab])
        # stack all neurons across probes
        neus3 = np.hstack(neus)
        Y = neus3.T

        M2 = np.zeros(Y.shape)
        if norm_:
            # normalise each neuron in each psth separately
            for i in range(M2.shape[0]):
                for psth in range(2):
                    bsl = np.mean(Y[i,75*psth:75*psth+25])
                    M2[i,75*psth:75*psth+75]=(
                        (Y[i,75*psth:75*psth+75]-bsl)/(bsl+0.5))
        else:
            M2 = Y

        pca = PCA()
        pca.fit(M2)
        u, s, vh = np.linalg.svd(M2)
        print('comps:', pca.n_components_, 'features:', pca.n_features_)
        print(pca.explained_variance_ratio_[:3])

        S_star = np.zeros(M2.shape)
        for i in range(2):
            S_star[i,i] = s[i]
        Y_re_star = np.matrix(u) * np.matrix(S_star) * np.matrix(vh)

        D[reg] = {'ns':ns,'labs':labs,
                  'Y':M2,'var_exp':pca.explained_variance_ratio_[:3],
                  'Y_re_star':Y_re_star}

    np.save(f'/home/mic/repro_manifold/single_cell_{split}2.npy', D,
            allow_pickle=True)
    print(errs)
    #return D


def multi_eid_PCA_dist(duration=1.5, lag=-0.5):

    '''
    choice trajectories of several sessions in the same plot
    to compare distribution
    '''
    sess = get_repeated_sites()
    regs = ['PPC', 'CA1', 'DG', 'LP', 'PO', None]
    for reg in regs:
        cs = {}
        for s in sess:
            eid = s[0]
            probe = s[1]
            try:
                w = PSTH(eid, duration, lag,probe=probe, reg=reg)
                ps = [np.mean(x,axis=1) for x in w]

                neus = np.concatenate(w)
                pca = PCA()
                pca.fit(neus)
                c = (pca.transform(neus)[:,:3]).T

                tra = trajectory_distance_simp(c, make_fig = False)

                cs[eid] = [ps,tra]
                gc.collect()
            except:
                print(f'something off with {eid}')
                cs[eid] = None
                gc.collect()
                continue

        np.save(f'/home/mic/repro_manifold/psth_d_new_{reg}.npy', cs,
                 allow_pickle=True)

        #return cs


'''########################
Plotting
########################'''


def psths_diff(pid,w=None,norm_=True, sfig=True):
    if sfig:
        plt.ioff()
    else:
        plt.ion()

    eid, pname = one.pid2eid(pid)


    #eid, pname = ['ebe090af-5922-4fcd-8fc6-17b8ba7bad6d', 'probe01']

    if w is None:
        w = PSTH(eid,probe=pname)

    fig = plt.figure(figsize=(10,6))
    axs = []

    duration = 1.5 #sec
    lag = -0.5

    minY = min([np.amin(y) for y in w])
    maxY = max([np.amax(y) for y in w])

    k = 0

    xs = np.linspace(0,int(duration/T_BIN),5)

    s = ' '.join([str(one.eid2path(eid)).split('/')[i] for i in [4,6,7]])

    age = get_probe_age(pid)

    axs.append(plt.subplot(1,3,k+1))

    d = w[0].T
    d_l = np.zeros(d.shape)
    if norm_:
        # normalise each neuron in each psth separately
        for i in range(d_l.shape[0]):
            bsl = np.mean(d[i,0:int(abs(lag)/T_BIN)])
            d_l[i]=(d[i]-bsl)/(bsl+0.5)
    else:
        d_l = d

    axs[k].imshow(d_l, cmap='Greys',aspect="auto",#'Greys'
        interpolation='none',vmin=minY,vmax=maxY)

    axs[k].set_xticks(xs)
    axs[k].set_xticklabels(xs*T_BIN + lag)
    axs[k].set_title(f'left choice PSTH ')
    axs[k].set_xlabel(f'time [sec], T_BIN={T_BIN} sec')
    axs[k].set_ylabel('neurons')

    axs[k].axvline(x=abs(lag/T_BIN), linewidth=0.5, linestyle='--',
                           c='g',label='stim on')

    axs[k].legend(loc='lower left')

    k+=1

    d = w[1].T
    d_r = np.zeros(d.shape)
    if norm_:
        # normalise each neuron in each psth separately
        for i in range(d_r.shape[0]):
            bsl = np.mean(d[i,0:int(abs(lag)/T_BIN)])
            d_r[i]=(d[i]-bsl)/(bsl+0.5)
    else:
        d_r = d

    axs.append(plt.subplot(1,3,k+1))#
    axs[k].imshow(d_r, cmap='Greys',aspect="auto",#'Greys'
        interpolation='none',vmin=minY,vmax=maxY)

    axs[k].set_xticks(xs)
    axs[k].set_xticklabels(xs*T_BIN + lag)
    axs[k].set_title(f'right choice PSTH')
    axs[k].set_xlabel(f'time [sec], T_BIN={T_BIN} sec')
    axs[k].set_ylabel('neurons')

    axs[k].axvline(x=abs(lag/T_BIN), linewidth=0.5, linestyle='--',
                           c='g',label='stim on')

    #axs[k].legend()

    k+=1
    # plot psth difference
    axs.append(plt.subplot(1,3,k+1))

    axs[k].imshow(abs(d_r - d_l), cmap='Greys',aspect="auto",#'Greys'
        interpolation='none',vmin=minY,vmax=maxY)

    axs[k].set_xticks(xs)
    axs[k].set_xticklabels(xs*T_BIN + lag)
    axs[k].set_title(f'abs(psth(left) - psth(right)) ')
    axs[k].set_xlabel(f'time [sec], T_BIN={T_BIN} sec')
    axs[k].set_ylabel('neurons')

    axs[k].axvline(x=abs(lag/T_BIN), linewidth=0.5, linestyle='--',
                           c='g',label='stim on')
    #axs[k].legend()

    ax2=axs[k].twinx()

    ax2.plot(np.mean(abs(d_r - d_l),axis=0),color='b',
             linewidth=0.5)
    ax2.set_ylabel('mean(abs(psth(left) - psth(right)))')
    ax2.yaxis.label.set_color('b')
    ax2.spines["right"].set_edgecolor('b')
    ax2.tick_params(axis='y', colors='b')
    k+=1

    plt.suptitle(f'{s}, probe age in days: {age}')
    plt.tight_layout()

    if sfig:
        plt.savefig(f'/home/mic/repro_manifold/'
                    f'no_response_cases/pid_{pid}.png')
        plt.close()
    #return w


def plot_cell_heatmaps(reg,share_ax=True, rmv_grey=False,athr = 0.2):

    '''
    plot psth per lab for a given region
    and corresponding Y_reconstr matrix
    '''
    #regs = ['PPC', 'CA1', 'DG', 'LP', 'PO', None]
#    s = '/home/mic/repro_manifold/single_cell_new.npy'
#    s = '/home/mic/repro_manifold/single_cell.npy'

    align = 'motion'
    if align == 'motion':
        s = '/home/mic/repro_manifold/single_cell_motion.npy'
    else:
        s = '/home/mic/repro_manifold/single_cell_new_norm.npy'

    D = np.load(s, allow_pickle=True).flat[0]

    Labs0 = ['mrsicflogellab','mainenlab','hoferlab','angelakilab',
            'churchlandlab','danlab','zadorlab','churchlandlab_ucla',
            'cortexlab','wittenlab']

    _,b,cols = labs_maps()

    Labs1 = [b[x] for x in Labs0]

    ns = D[reg]['ns']
    labs = [b[x] for x in D[reg]['labs']]
    UV = D[reg]['Y_re_star'] # that was UV but now is reconstructed Y
    Y = D[reg]['Y']


    # Exclude labs with less than 4 recordings
    r = Counter(labs)
    Labs = [x for x in Labs1 if r[x]>3]

    bds = dict(zip(list(Counter(labs)),[[] for _ in range(len(Counter(labs)))]))
    for i in range(len(labs)):
        bds[labs[i]].append(ns[i])

    for i in bds:
        bds[i].insert(0,0)
        bds[i] = np.cumsum(bds[i])

    ns.insert(0,0)
    ns = np.cumsum(ns)


    # cut the stack Y and UV into labs again

    centsUV = dict(zip(Labs,[[] for _ in range(len(Labs))]))
    centsY = dict(zip(Labs,[[] for _ in range(len(Labs))]))

    for i in range(len(labs)):
        if labs[i] not in Labs:
            continue

        Y1 = Y[ns[i]:ns[i+1]]
        U1 = UV[ns[i]:ns[i+1]]
        centsUV[labs[i]].append(U1)
        centsY[labs[i]].append(Y1)

    cents2UV = {lab:np.concatenate(centsUV[lab])
              for lab in centsUV if centsUV[lab]!=[]}

    minUV = min([np.amin(cents2UV[lab])for lab in cents2UV
            if cents2UV[lab].size != 0])
    maxUV = max([np.amax(cents2UV[lab]) for lab in cents2UV
            if cents2UV[lab].size != 0])

    cents2Y = {lab:np.concatenate(centsY[lab])
              for lab in centsY if centsY[lab]!=[]}

    minY = min([np.amin(cents2Y[lab]) for lab in cents2Y
            if cents2Y[lab].size != 0])
    maxY = max([np.amax(cents2Y[lab]) for lab in cents2Y
            if cents2Y[lab].size != 0])

    if share_ax:
        # match color scale
        minUV = minY
        maxUV = maxY
    else:
        maxY = None
        minY = None

    fig = plt.figure(figsize=(10,7))
    Ax = []
    k = 0

    l_abs = list(cents2Y.keys())
    ss = [cents2Y[lab].shape[0] for lab in l_abs]
    l_abs = list(np.array(l_abs)[np.argsort(ss)])
    l_abs.reverse()
    ss = [cents2Y[lab].shape[0] for lab in l_abs]

    ss.insert(0,0)
    ss = np.cumsum(ss)
    ss = ss[:-1]

    # stack again into n chunks
    nchunks = 5

    M_all = np.vstack([np.hstack([cents2Y[lab], cents2UV[lab]])
                  for lab in l_abs])

    sss = [ss[i] + bds[l_abs[i]][1:-1] for i in range(len(ss))]
    sss_flat = [x for y in sss for x in y]

    M_all = np.array(M_all)

    print(f'data dims: {M_all.shape}')
    if rmv_grey: # remove units with weak activity change

        #athr = 0.2
        M_red = []
        sss_flat_rmv = []
        ss_rmv = []
        k = 0
        for i in range(M_all.shape[0]):
            if max(abs(M_all[i])) > athr:
                M_red.append(M_all[i])
                k += 1
            if i in ss:
                ss_rmv.append(k)
            if i in sss_flat:
                sss_flat_rmv.append(k)

        M_all = np.array(M_red)
        sss_flat = sss_flat_rmv
        ss = ss_rmv
        print(f'data dims after rmv_grey: {M_all.shape}')

    u = M_all.shape[0]//nchunks

    rc = 0
    rcc = 0
    k = 0
    for chunk in range(nchunks):

        if k != 0:
            Ax.append(plt.subplot(1,nchunks,k+1))#,sharey=Ax[0]))
            Ax[k].get_yaxis().set_visible(False)
            Ax[k].set_xlabel('time')

        else:
            Ax.append(plt.subplot(1,nchunks,k+1))
            Ax[k].set_xlabel('time [sec]')
#        Ax[k].set_anchor('N')

        Ax[k].set_title(f'{chunk+1}/{nchunks}')

        M = M_all[u*chunk:u*(1+chunk)]

        Ax[k].imshow(M, cmap='Greys',aspect="auto",#'Greys'
        interpolation='none',vmin=minY,vmax=maxY)

        for x in [25, 100, 175, 250]:
            Ax[k].axvline(x=x, linewidth=0.5, linestyle='--',
                                   c='g',label='stim on')
            if rcc == 0:
                Ax[k].text(0,u/2,'stim on',rotation=90,
                           color='g')
            rcc +=1


        Ax[k].axvline(x=150, linewidth=1, linestyle='-',
                               c='k',label='separate Y and Y_re')

        for x in [75, 225]:
            Ax[k].axvline(x=x, linewidth=0.5, linestyle='-',
                                   c='k',label='separate left and right')
            if k == 0:
                if x == 75:
                    s = 'separate left and right choice PETHs'
                else:
                    s = 'separate left and right choice 2-pc-reconstructed PETHs'
                Ax[k].text(x - 25,0.9 * u,s,
                           rotation=90,color='k')


        # plot lab boundaries
        for b in ss:
            if b in range(u*chunk,u*(1+chunk)):
                Ax[k].axhline(y=b-u*chunk, linewidth=1, linestyle='-',
                              c='b',label='lab boundary')
                Ax[k].text(0,b-u*chunk,l_abs[list(ss).index(b)])


         # plot session boundaries

        for b in sss_flat:
            if b in range(u*chunk,u*(1+chunk)):
                Ax[k].axhline(y=b-u*chunk, linewidth=0.5, linestyle='--',
                              c='r',label='session boundary')
                if rc == 0:
                    Ax[k].text(0,b-u*chunk,'session boundary',
                               color='r')

                rc +=1


        Ax[k].set_xticks(np.linspace(0,300,5))
        if k == 0:
            Ax[k].set_xticklabels([0,1.5,'','',''])
        else:
            Ax[k].set_xticklabels(['','','','',''])


#        if k != 0:
#            yt = Ax[0].get_yticklabels()
#            ytn = [str(w.get_position()[1]+u*chunk) for w in yt]
#            Ax[k].set_yticklabels(ytn)

        if k == 0:
            Ax[k].set_ylabel('neurons')
        k += 1


    plt.tight_layout()
    if reg is None:
        reg = 'whole probe'
    #plt.suptitle(f'PSTH left choice | PSTH right choice || PSTH via 2 PCs left | PSTH via 2 PCs right; Region: {reg}; share_ax={share_ax}; pnorm=True')
    plt.tight_layout()#rect=[0, 0.03, 1, 0.95])
#    plt.savefig(f'/home/mic/repro_manifold/heatmaps_new/'
#                f'heatmaps_{reg}_norm.png')
#    plt.close()



def plot_single_cell(new_spike=True):

#    s = '/home/mic/repro_manifold/single_cell_new.npy'
#    s = '/home/mic/repro_manifold/single_cell.npy'


    s = '/home/mic/repro_manifold/single_cell_motion.npy'

    D = np.load(s, allow_pickle=True).flat[0]

    _,b,lab_cols = labs_maps()

    fig = plt.figure(figsize=(10,5))
    Ax = []

    k = 0
    for reg in D:
        if k == 0:
            Ax.append(plt.subplot(2,3,k+1))
            Ax[0].set_xlim([-2, 2])
            Ax[0].set_ylim([-2, 2])
        else:
            Ax.append(plt.subplot(2,3,k+1, sharex=Ax[0],sharey=Ax[0]))

        if k%3 != 0:
            Ax[k].get_yaxis().set_visible(False)
        if k < 3:
            Ax[k].get_xaxis().set_visible(False)

        ns = D[reg]['ns']
        labs = [b[x] for x in D[reg]['labs']]
        Y = D[reg]['Y']

        pca = PCA()
        UV = pca.fit_transform(Y)

        ns.insert(0,0)
        ns = np.cumsum(ns)

        r = Counter(labs)
        Labs = [x for x in r if r[x]>3]

        cents = dict(zip(Labs,[ [] for _ in range(len(Labs))]))

        for i in range(len(labs)):

            if labs[i] not in Labs:
                continue

            U1 = UV[ns[i]:ns[i+1],:2]
            Ax[k].scatter(U1[:,0],U1[:,1],color=lab_cols[labs[i]],
                          s=4,marker='.')

            cents[labs[i]].append(U1)

        cents2 = {lab:np.mean(np.concatenate(cents[lab]),axis=0)
                  for lab in cents if cents[lab]!=[]}

        # plot means
        for lab in cents2:
            Ax[k].scatter(cents2[lab][0],cents2[lab][1],
                          s=400,marker='x',color=lab_cols[lab])

            # plot confidence ellipses
        for lab in cents2:
            x = np.concatenate(cents[lab])
            confidence_ellipse(x[:,0], x[:,1], Ax[k], n_std=1.0,
                               edgecolor=lab_cols[lab])


        varex = str(np.round(D[reg]['var_exp'][:2],2))
        if reg is None:
            reg = 'whole probe'
        Ax[k].set_title(f"{reg}; Var. expl.:{varex}")
        Ax[k].set_xlabel('pc0')
        Ax[k].set_ylabel('pc1')
        k += 1

    legend_elements = [Patch(facecolor=lab_cols[lab],
                       edgecolor=lab_cols[lab],
                       label=lab) for lab in Labs]

    plt.tight_layout()
#    plt.subplots_adjust(top=0.845,
#                        bottom=0.11,
#                        left=0.11,
#                        right=0.9,
#                        hspace=0.29,
#                        wspace=0.33)
    #plt.suptitle(f'new_spike={new_spike}; pnorm={pnorm}')
    Ax[0].legend(handles=legend_elements,
     loc='upper left',bbox_to_anchor=(0, 0.5), ncol=5).set_draggable(True)



def average_dist_plot(psth=False, norm_=True,
                      axoff=False,new_spike=True):

    '''
    Grid plot, #regions x # labs
    One distance line per recording,
    superimpose recordings per lab

    show distance of choice manifolds or psths directly
    '''

    one = ONE()

    Cs = {}
    regs = ['PPC', 'CA1', 'DG', 'LP', 'PO', None]
    for reg in regs:
        s = f'/home/mic/repro_manifold/psth_d_new_{reg}.npy'
        Cs[reg] = np.load(s, allow_pickle=True).flat[0]

    _,b,cols = labs_maps()

    d_ = Counter([b[str(one.eid2path(eid)).split('/')[4]]
                 for eid in Cs[None] if
                 (Cs[None][eid] is not None)])

    labs_ = [x for x in d_ if d_[x]>3]

    fig = plt.figure(figsize=(12,10))


    ax0 = plt.subplot(2,3,1)
    axs = [plt.subplot(2,3,k,sharex=ax0,sharey=ax0)
           for k in range(2,len(regs)+1)]
    axs.insert(0,ax0)
    axs = np.array(axs)

    k = 0
    for reg in regs:

        axs[k].axvline(x=0, linewidth=2,
             linestyle='--')

        for eid in Cs[reg]:


                lab = b[str(one.eid2path(eid)).split('/')[4]]

                if (Cs[reg][eid] is None) or (lab not in labs_):
                    continue

                if psth:
                    ys = abs(Cs[reg][eid][0][0] - Cs[reg][eid][0][1])
                else:
                    ys = Cs[reg][eid][1]
                if norm_:
                    bsl = np.mean(ys[:25])
                    ys = (ys - bsl)/(bsl + 0.5)

                else:
                    ys = zscore(ys)

                xs = np.arange(len(ys))*T_BIN - 0.5
                axs[k].plot(xs,ys,color=cols[lab],alpha=0.5)


                if reg == None:
                    regt = 'whole probe'
                else:
                    regt = reg
                axs[k].set_title(regt,fontsize=10)
        k += 1

    legend_elements = [Patch(facecolor=cols[lab], edgecolor=cols[lab],
                       label=lab) for lab in labs_]

    plt.legend(handles=legend_elements,
               loc='upper left',bbox_to_anchor=(0, 1),
               ncol=2).set_draggable(True)

    plt.suptitle(f'left choice - right choice; psth={psth}')
    plt.tight_layout()
#    plt.savefig(f'/home/mic/repro_manifold/'
#                f'psth_mani_dist/psth_{psth}_sharex{share_ax}_new.png')


def grid_PCA_plot_by_lab(reg=None, dim2=True, dist_=True):

    '''
    for a region in this list, or reg=None (all neurons of probe)
    show for each recording the activity distance (dist=True)
    or the 2d state space; choice trajectories
    '''

    # regs of interest: ['PPC', 'CA1', 'DG', 'LP', 'PO']
    one = ONE()
    cs = np.load(f'/home/mic/repro_manifold/pcs_{reg}.npy',
                 allow_pickle=True).flat[0]
    _,b,cols = labs_maps()
    d = Counter([b[str(one.eid2path(eid)).split('/')[4]]
                 for eid in cs if
                 ((cs[eid] is not None) and (cs[eid].shape[0] == 3))])
    rows = dict(zip(d.keys(),range(len(d.keys()))))
    ncols = max(d.values())
    print('number of columns', ncols)
    rcs = dict(zip(d.keys(),[0]*len(d)))

    fig = plt.figure(figsize=(12,10))

    for eid in cs:

        if (cs[eid] is None) or (cs[eid].shape[0] != 3):
            continue

        lab = b[str(one.eid2path(eid)).split('/')[4]]
        row = rows[lab]
        column = rcs[lab]
        if dist_:
            dim2=True

        if dim2:
            ax = plt.subplot(len(d),ncols, ncols*row + column + 1)
        else:
            ax = plt.subplot(len(d),ncols, ncols*row + column + 1,projection='3d')



        if dist_:
            trajectory_distance_simp(cs[eid], ax=ax)
            ax.axis('off')
        else:
            scat3d(cs[eid],dim2=dim2,ax=ax)
            ax.axis('off')

            @interact(dist=(1, 20, 1))
            def update(dist=1):
                ax.dist = dist
                display(fig)

        plt.title(' '.join(np.array(str(one.eid2path(eid)).split('/'))[[6,7]]),
                  fontsize=10)
        rcs[lab] += 1

    if reg is None:
        reg = 'all clusters of probe'
    plt.suptitle(reg)
    plt.tight_layout()
    plt.savefig(f'/home/mic/repro_manifold/overview_plots_trajectories3d/{reg}.png')


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def scat3d(c, choice_split=True, lag=-0.5, ax=None, dim2 = False):

    '''
    MAIN TRAJECTORY PLOTTING
    '''

    if dim2:
        if ax == None:
            fig,ax = plt.subplots()
    else:
        if ax == None:
            fig = plt.figure(figsize=(3,3))
            ax = fig.add_subplot(111, projection='3d')

    bipars = 0
    n_traj = 2
    win = int(len(c[0])/n_traj)
    cbar_types = ['Blues','Reds'] + ['Greys' for i in range(bipars*2)]

    if choice_split:
        labels = ['choice left','choice right', f'{bipars} x pseudo'] + [
                 '_nolegend_' for i in range(bipars*2 -1 )]
    else:
        labels = ['pleft 0.8','pleft 0.2', f'{bipars} x pseudo'] + [
                 '_nolegend_' for i in range(bipars*2 -1 )]

    k = 0
    for traj in range(n_traj):

        xs = c[0][win * traj:win * (traj + 1)]
        ys = c[1][win * traj:win * (traj + 1)]
        zs = c[2][win * traj:win * (traj + 1)]
        Bc = np.arange(len(xs))
        cmap = plt.get_cmap(cbar_types[k])(np.linspace(0,1,20))
        cm = mpl.colors.ListedColormap(cmap[10:,:-1])


        if dim2:
            ax.scatter(xs,ys, c=Bc,cmap=cm)

        else:
            ax.scatter(xs,ys,zs, c=Bc,cmap=cm, depthshade=False)


            ax.plot(xs,ys,zs, color='k', linewidth = 0.2)

        k += 1

    if dim2:
        ax.set_xlabel('pc1')
        ax.set_ylabel('pc2')

    else:
        ax.set_xlabel('pc1')
        ax.set_ylabel('pc2')
        ax.set_zlabel('pc3')
        set_axes_equal(ax)


    legend_elements = [Patch(facecolor='b', edgecolor='b',
                       label='pleft0.2'),
                       Patch(facecolor='r', edgecolor='r',
                       label='pleft0.8')]

#    plt.legend(handles=legend_elements,
#               loc='upper left',bbox_to_anchor=(0, 1), ncol=2)
    #plt.tight_layout()


def trajectory_distance_simp(c, make_fig = True, lag=-0.5, ax=None,
                        verbose = False, choice_split=True, norm_=True):

    '''
    compute and display the distance between choice
    trajectories;
    input: c
    '''

    n_traj = 2
    win = int(len(c[0])/n_traj)

    labels = ['d choice left-right']

    k = 0

    XYZs = []
    for traj in range(n_traj):

        xs = c[0][win * traj:win * (traj + 1)]
        ys = c[1][win * traj:win * (traj + 1)]
        zs = c[2][win * traj:win * (traj + 1)]
        XYZs.append([xs, ys, zs])

    XYZs = np.array(XYZs)

    def distE(x,y):
        return np.sqrt(np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y))

    # compute distance of trajectories for each pair, real and pseudo
    d_01 = [distE(XYZs[0][:,i], XYZs[1][:,i]) for i in range(win)]
    D = {'distance of block traj.':d_01}
    cols = {'distance of block traj.': 'k'}

    xs = np.arange(win)*T_BIN

    if make_fig:
        if ax == None:
            fig,ax = plt.subplots(figsize=(3,3))

        # plot distance left right
        k = 0
        for d in D:
            if norm_:
                ys = D[d] - np.mean(D[d][:int(lag/T_BIN)])
            else:
                ys = D[d]
            ax.plot(xs,ys, label = labels[k],
                    c = cols[d], linewidth=1, alpha=0.5)
            k+=1


        if lag <= 0:
            ax.axvline(x=abs(lag), linewidth=2, linestyle='--',
                       c='g',label='stim on')

        #plt.legend(fontsize=7,ncol=1)
        plt.xlabel('time [sec]')
        #plt.ylabel('Euclidean distance \n of points in trajectories [a.u.]')
        plt.tight_layout()

    else:

#        if norm_:
#            d = 'distance of block traj.'
#            ys = D[d] - np.mean(D[d][:int(lag/T_BIN)])

        return d_01


def illustrate_reconstruction(reg):

    '''
    check mosaic plot function to get different space allocation
    '''


    s = '/home/mic/repro_manifold/single_cell_new_norm.npy'
    D = np.load(s, allow_pickle=True).flat[0][reg]

    # determine goodness of reconstruction via r2 score
    y = D['Y']
    y_re = np.array(D['Y_re_star'])

    # get lab per neuron
    ns = D['ns']
    labs = D['labs']
    ns.insert(0,0)
    ns = np.cumsum(ns)

    athr = 0.2

    r2s = []
    l2s = []
    iex = []
    for i in range(len(y)):
        if abs(max(y[i])) > athr:
            r2s.append(r2_score(y[i],y_re[i]))
            l2s.append(np.sum(np.power((y[i]-y_re[i]),2)))
            iex.append(i)


    fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(5,3))
    # illustrate example cells
    # 3 with great, 3 with medium, 3 with bad reconstruction
    t =np.argsort(r2s)
    t_ = np.concatenate([t[:1],t[-1:]])#,t[len(t)//2:len(t)//2 +1]

    xs = np.arange(len(y[0]))*T_BIN

    for k in [iex[j] for j in t_]:

        axs[0].plot(xs,y[k]/T_BIN,color='k')
        axs[0].plot(xs,y_re[k]/T_BIN,color='r',linestyle='--')

    for x in np.array([25, 100])*T_BIN:
        axs[0].axvline(x=x, linewidth=0.5, linestyle='--',
                               c='g',label='stim on')

    axs[0].set_xlabel('time [sec]')
    axs[0].set_ylabel('firing rate [Hz]')
    axs[0].set_title('example cells PSTHs \n and PCA-reconstruction')


    n, bins, patches = axs[1].hist(x=r2s, bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)

    axs[1].set_xlabel(r'$r^2$')
    axs[1].set_ylabel('number of neurons')
    axs[1].set_title('goodness of PSTH \n reconstruction')

    plt.tight_layout()


def mscatter(x,y,ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    if not ax: ax=plt.gca()
    sc = ax.scatter(x,y,**kw)
    if (m is not None) and (len(m)==len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc


def single_cell_reg_vs_lab(algo = 'PCA',rm_unre=True,
                           shuf=False,align='motion',
                           lim5 = True):

    # may also try umap here instead of PCA!

    _,b,lab_cols = labs_maps()

    if align == 'motion':
        s = '/home/mic/repro_manifold/single_cell_motion.npy'
    else:
        s = '/home/mic/repro_manifold/single_cell_new_norm.npy'
    D = np.load(s, allow_pickle=True).flat[0]

    ys = []
    regs_ = []
    labs_ = []
    for reg in D:
        if reg is None:
            continue

        ys.append(D[reg]['Y'])
        regs_.append([reg]*(D[reg]['Y'].shape[0]))
        # get lab and region per neuron
        ns = D[reg]['ns']
        labs = D[reg]['labs']
        assert len(ns) == len(labs), 'ns/labs length mismatch!'
        for i in range(len(labs)):
            labs_.append([labs[i]]*ns[i])

    y = np.concatenate(ys)
    regs = np.concatenate(regs_)
    labs = np.concatenate(labs_)

    xs = np.arange(len(y[0]))*T_BIN

    if shuf:
        np.random.shuffle(labs)

    lab3 = ['wittenlab','zadorlab','cortexlab']

    if rm_unre:
        # remove non-responsive units
        athr = 0.2

        bad = []
        for i in range(len(y)):
            if abs(max(y[i])) < athr:
                bad.append(i)
                continue
            if lim5: # restrict to
                if labs[i] in lab3:
                    bad.append(i)

        y = np.delete(y,bad,axis=0)
        #y = np.array(y)
        regs = np.delete(regs,bad,axis=0)
        labs = np.delete(labs,bad,axis=0)

    # merge hofer and mirsicflogel labs
    labs[labs=='mrsicflogellab'] = 'hoferlab'

    if algo == 'umap':
        emb = umap.UMAP(random_state=8).fit_transform(y)
    if algo == 'tSNE':
        emb = TSNE(n_components=2,perplexity=30,
                          random_state=8).fit_transform(y)
    if algo == 'PCA':
        pca = PCA(n_components=2)
        pca.fit(y)
        emb = pca.transform(y)


    #fig,axs = plt.subplots(nrows=2,ncols=2,figsize=(5,6))

    fig = plt.figure()

    mosaic = """
            AB
            CD
            """
    axs = fig.subplot_mosaic(mosaic)



    cols_lab = [lab_cols[b[x]] for x in labs]
    axs['A'].scatter(emb[:,0],emb[:,1],marker='o',c=cols_lab, s=2)

    labs_ = Counter(labs)

    cents = {}
    for lab in labs_:
        cents[lab] = np.mean(emb[labs == lab],axis=0)

    # plot means
    for lab in cents:
        axs['A'].scatter(cents[lab][0],cents[lab][1],
                      s=800,marker='x',color=lab_cols[b[lab]])

    # plot confidence ellipses
    for lab in labs_:
        x = emb[labs == lab]
        confidence_ellipse(x[:,0], x[:,1], axs['A'], n_std=1.0,
                           edgecolor=lab_cols[b[lab]])


    le = [Patch(facecolor=lab_cols[b[lab]],
           edgecolor=lab_cols[b[lab]],
           label=b[lab]) for lab in labs_]

    axs['A'].legend(handles=le,
     loc='upper left',bbox_to_anchor=(0, 0.5), ncol=1).set_draggable(True)

    axs['A'].set_xlabel('embedding dim 1')
    axs['A'].set_ylabel('embedding dim 2')
    plt.tight_layout()

    # plot average PSTHs across labs
    for lab in labs_:
        ms = np.mean((y[labs == lab])/T_BIN,axis=0)
        ste = np.std((y[labs == lab])/T_BIN,axis=0)/np.sqrt(len(y[labs == lab]))

        axs['C'].plot(xs,ms,color=lab_cols[b[lab]])
        axs['C'].fill_between(xs, ms + ste, ms - ste,
                              color=lab_cols[b[lab]],alpha=0.2)



    axs['C'].set_xlabel('time [sec]')
    axs['C'].set_ylabel('firing rate [Hz]')
    for x in np.array([25, 100])*T_BIN:
        axs['C'].axvline(x=x, linewidth=0.5, linestyle='--',
                               c='g',label='motion start')

    D = reg_cols()
    regs_ = Counter(regs)
    cols_reg = [D[x] for x in regs]
    axs['B'].scatter(emb[:,0],emb[:,1],marker='o',c=cols_reg, s=2)

    regs_ = Counter(regs)
    cents = {}
    for reg in regs_:
        cents[reg] = np.mean(emb[regs == reg],axis=0)

    # plot means
    for reg in cents:
        axs['B'].scatter(cents[reg][0],cents[reg][1],
                      s=800,marker='x',color=D[reg])

    # plot confidence ellipses
    for reg in regs_:
        x = emb[regs == reg]
        confidence_ellipse(x[:,0], x[:,1], axs['B'],
                           n_std=1.0,edgecolor=D[reg])


    le = [Patch(facecolor=D[reg],
           edgecolor=D[reg],
           label=reg) for reg in regs_]

    plt.legend(handles=le,
     loc='upper left',bbox_to_anchor=(0, 0.5), ncol=1).set_draggable(True)

    axs['B'].set_xlabel('embedding dim 1')
    axs['B'].set_ylabel('embedding dim 2')

    axs['B'].sharex(axs['A'])
    axs['B'].sharey(axs['A'])

    axs['C'].sharex(axs['D'])
    axs['C'].sharey(axs['D'])

    # plot average PSTHs
    for reg in regs_:
        ms = np.mean((y[regs == reg])/T_BIN,axis=0)
        ste = np.std((y[regs == reg])/T_BIN,axis=0)/np.sqrt(len(y[regs == reg]))

        axs['D'].plot(xs,ms,color=D[reg])
        axs['D'].fill_between(xs, ms + ste, ms - ste,
                              color=D[reg],alpha=0.2)

    for x in np.array([25, 100])*T_BIN:
        axs['D'].axvline(x=x, linewidth=0.5, linestyle='--',
                               c='g',label='motion start')
    axs['D'].set_xlabel('time [sec]')
    axs['D'].set_ylabel('firing rate [Hz]')

    plt.suptitle(f'{algo}; dim reduction of PSTH; {len(y)} clusters; \n lab name shuffle {shuf}; responsive only {rm_unre}; align = {align}')
    plt.tight_layout()


def reg_cols():

    return {'PPC': sns.color_palette('colorblind')[0],
        'CA1': sns.color_palette('colorblind')[2],
        'DG': sns.color_palette('muted')[2],
        'LP': sns.color_palette('colorblind')[4],
        'PO': sns.color_palette('colorblind')[6],
        'RS': sns.color_palette('Set2')[0],
        'FS': sns.color_palette('Set2')[1],
        'RS1': sns.color_palette('Set2')[2],
        'RS2': sns.color_palette('Set2')[3]}


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std# /np.sqrt(len(x))
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std# /np.sqrt(len(y))
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)



def all_panels(algo = 'PCA',rm_unre=True,
               align='motion',lim5 = True,split='RT'):

    # may also try umap here instead of PCA!

    _,b,lab_cols = labs_maps()

    if split == 'choice':
        s = '/home/mic/repro_manifold/single_cell_motion.npy'
        ts = 'left|right choice PSTH'
    if split == 'RT':
        s = '/home/mic/repro_manifold/single_cell_RT2.npy'
        ts = 'fast|slow RT PSTH'


    D = np.load(s, allow_pickle=True).flat[0]

    ys = []
    ys_re = []
    regs_ = []
    labs_ = []
    for reg in D:
        if reg is None:
            continue

        ys.append(D[reg]['Y'])
        ys_re.append(np.array(D[reg]['Y_re_star']))
        regs_.append([reg]*(D[reg]['Y'].shape[0]))
        # get lab and region per neuron
        ns = D[reg]['ns']
        labs = D[reg]['labs']
        assert len(ns) == len(labs), 'ns/labs length mismatch!'
        for i in range(len(labs)):
            labs_.append([labs[i]]*ns[i])

    y = np.concatenate(ys)
    y_re = np.concatenate(ys_re)
    regs = np.concatenate(regs_)
    labs = np.concatenate(labs_)


    xs = np.arange(len(y[0]))*T_BIN

    lab3 = ['wittenlab','zadorlab','cortexlab']

    if rm_unre:
        # remove non-responsive units
        athr = 0.2

        bad = []
        for i in range(len(y)):
            if abs(max(y[i])) < athr:
                bad.append(i)
                continue
            if lim5: # restrict to
                if labs[i] in lab3:
                    bad.append(i)

        print(y.shape, y_re.shape, len(bad))
        y = np.delete(y,bad,axis=0)
        y_re = np.delete(y_re,bad,axis=0)
        regs = np.delete(regs,bad,axis=0)
        labs = np.delete(labs,bad,axis=0)


    # merge hofer and mirsicflogel labs
    labs[labs=='mrsicflogellab'] = 'hoferlab'

    if algo == 'umap':
        emb = umap.UMAP(random_state=8).fit_transform(y)
    if algo == 'tSNE':
        emb = TSNE(n_components=2,perplexity=30,
                          random_state=8).fit_transform(y)
    if algo == 'PCA':
        pca = PCA(n_components=2)
        pca.fit(y)
        emb = pca.transform(y)

    #fig,axs = plt.subplots(nrows=2,ncols=2,figsize=(5,6))

    fig = plt.figure(figsize=(16,10))

    inner = [['Ea'],
             ['Eb']]

    mosaic = [['C','A','Ia','I',None],
              ['Ga','G','Ja','J',None],
              ['Ha','H','Ka','K',None],
              ['D','B','L',inner,'F']]

    ms2 = ['Ga','Ha','Ia','Ja','Ka']
    #panel_n = {


    axs = fig.subplot_mosaic(mosaic)

    cols_lab = [lab_cols[b[x]] for x in labs]
    axs['A'].scatter(emb[:,0],emb[:,1],marker='o',c=cols_lab, s=2)

    labs_ = Counter(labs)

    cents = {}
    for lab in labs_:
        cents[lab] = np.mean(emb[labs == lab],axis=0)

    # plot means
    for lab in cents:
        axs['A'].scatter(cents[lab][0],cents[lab][1],
                      s=800,marker='x',color=lab_cols[b[lab]])

    # plot confidence ellipses
    for lab in labs_:
        x = emb[labs == lab]
        confidence_ellipse(x[:,0], x[:,1], axs['A'], n_std=1.0,
                           edgecolor=lab_cols[b[lab]])

    # permutation test
    # for a given pair of labs there's a distance of means
    def distE(x,y):
        return np.sqrt(np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y))

    nrand = 20  #random lab allocations
    centsr = []
    for shuf in range(nrand):
        labsr = labs.copy()
        random.shuffle(labsr)
        cenr = {}
        for lab in labs_:
            cenr[lab] = np.mean(emb[labsr == lab],axis=0)
        centsr.append(cenr)

    comb = combinations(cents, 2)

    ps = {}
    for pair in comb:
        dist = distE(cents[pair[0]],cents[pair[1]])
        null_d = [distE(cenr[pair[0]],cenr[pair[1]]) for cenr in centsr]
        p = 1 - (0.01 * percentileofscore(null_d,dist))
        ps[pair] = p


    le = [Patch(facecolor=lab_cols[b[lab]],
           edgecolor=lab_cols[b[lab]],
           label=b[lab]) for lab in labs_]

    axs['A'].legend(handles=le,
     loc='lower right', ncol=1).set_draggable(True)

    axs['A'].set_title('all cells')
    axs['A'].set_xlabel('embedding dim 1')
    axs['A'].set_ylabel('embedding dim 2')
    axs['A'].text(-0.1, 1.15, 'A', transform=axs['A'].transAxes,
      fontsize=16,  va='top', ha='right', weight='bold')


    # plot average PSTHs across labs
    for lab in labs_:
        ms = np.mean((y[labs == lab])/T_BIN,axis=0)
        ste = np.std((y[labs == lab])/T_BIN,axis=0)/np.sqrt(len(y[labs == lab]))

        axs['C'].plot(xs,ms,color=lab_cols[b[lab]])
        axs['C'].fill_between(xs, ms + ste, ms - ste,
                              color=lab_cols[b[lab]],alpha=0.2)




    axs['C'].set_title(ts)
    axs['C'].set_xlabel('time [sec]')
    axs['C'].set_ylabel('firing rate [Hz]')
    for x in np.array([25, 100])*T_BIN:
        axs['C'].axvline(x=x, lw=0.5, linestyle='--',
                               c='g',label='motion start')

    axs['C'].axvline(x=75*T_BIN, lw=2, linestyle='-',
                           c='cyan',label='trial cut')

    axs['C'].text(-0.1, 1.15, 'C', transform=axs['C'].transAxes,
      fontsize=16,  va='top', ha='right', weight='bold')

    le = [Line2D([0], [0], color='g', lw=0.5, ls='--', label='motion start'),
          Line2D([0], [0], color='cyan', lw=2, ls='-', label='trial cut')]

    axs['C'].legend(handles=le,
     loc='lower right', ncol=1).set_draggable(True)

    Dc = reg_cols()
    regs_ = Counter(regs)
    cols_reg = [Dc[x] for x in regs]
    axs['B'].scatter(emb[:,0],emb[:,1],marker='o',c=cols_reg, s=2)

    regs_ = Counter(regs)
    cents = {}
    for reg in regs_:
        cents[reg] = np.mean(emb[regs == reg],axis=0)

    # plot means
    for reg in cents:
        axs['B'].scatter(cents[reg][0],cents[reg][1],
                      s=800,marker='x',color=Dc[reg])

    # plot confidence ellipses
    for reg in regs_:
        x = emb[regs == reg]
        confidence_ellipse(x[:,0], x[:,1], axs['B'],
                           n_std=1.0,edgecolor=Dc[reg])

    le = [Patch(facecolor=Dc[reg],
           edgecolor=Dc[reg],
           label=reg) for reg in regs_]

    axs['B'].legend(handles=le,
     loc='lower right', ncol=1).set_draggable(True)

    axs['B'].set_title('all cells')
    axs['B'].set_xlabel('embedding dim 1')
    axs['B'].set_ylabel('embedding dim 2')
    axs['B'].text(-0.1, 1.15, 'B', transform=axs['B'].transAxes,
      fontsize=16,  va='top', ha='right', weight='bold')

    axs['B'].sharex(axs['A'])
    axs['B'].sharey(axs['A'])

    axs['C'].sharex(axs['D'])
    axs['C'].sharey(axs['D'])

    # plot average PSTHs
    for reg in regs_:
        ms = np.mean((y[regs == reg])/T_BIN,axis=0)
        ste = np.std((y[regs == reg])/T_BIN,axis=0)/np.sqrt(len(y[regs == reg]))

        axs['D'].plot(xs,ms,color=Dc[reg])
        axs['D'].fill_between(xs, ms + ste, ms - ste,
                              color=Dc[reg],alpha=0.2)

    for x in np.array([25, 100])*T_BIN:
        axs['D'].axvline(x=x, linewidth=0.5, linestyle='--',
                               c='g',label='motion start')

    axs['D'].axvline(x=75*T_BIN, lw=2, linestyle='-',
                           c='cyan',label='trial cut')

    axs['D'].set_title(ts)
    axs['D'].set_xlabel('time [sec]')
    axs['D'].set_ylabel('firing rate [Hz]')
    axs['D'].text(-0.1, 1.15, 'D', transform=axs['D'].transAxes,
      fontsize=16,  va='top', ha='right', weight='bold')

    le = [Line2D([0], [0], color='g', lw=0.5, ls='--', label='motion start'),
          Line2D([0], [0], color='cyan', lw=2, ls='-', label='trial cut')]

    axs['D'].legend(handles=le,
     loc='lower right', ncol=1).set_draggable(True)

    '''
    check mosaic plot function to get different space allocation
    '''

    r2s = []
    l2s = []
    iex = []
    for i in range(len(y)):
        r2s.append(r2_score(y[i],y_re[i]))
        l2s.append(np.sum(np.power((y[i]-y_re[i]),2)))
        iex.append(i)

    # illustrate example cells
    # 1 with great, 1 with bad reconstruction
    t = np.argsort(r2s)
    t_ = np.concatenate([t[170:171],t[-1:]])#,t[len(t)//2:len(t)//2 +1]

    xs = np.arange(len(y[0]))*T_BIN

    # split good and bad example cell into two panels
    ms = ['Ea','Eb']
    idxs = [iex[j] for j in t_]

    for k in range(len(idxs)):

        axs[ms[k]].plot(xs,y[idxs[k]]/T_BIN,c='k',label='PSTH')
        axs[ms[k]].plot(xs,y_re[idxs[k]]/T_BIN,c='r',ls='--',label='fit')

        for x in np.array([25, 100])*T_BIN:
            axs[ms[k]].axvline(x=x, linewidth=0.5, linestyle='--',
                                   c='g',label='motion start')
        axs[ms[k]].axvline(x=75*T_BIN, lw=2, linestyle='-',
                               c='cyan',label='trial cut')
        axs[ms[k]].set_ylabel('firing rate \n [Hz]')
        axs[ms[k]].text(1, 0.2,
                        rf'$r^2$={np.round(r2_score(y[idxs[k]],y_re[idxs[k]]),2)}',
                        transform=axs[ms[k]].transAxes,
                        fontsize=8,  va='top', ha='right')
        k+=1


    axs['Eb'].set_xlabel('time [sec]')

    axs['Ea'].set_title('example cells PSTHs \n and PCA-reconstruction')

    axs['Ea'].text(-0.1, 1.15, 'E', transform=axs['Ea'].transAxes,
      fontsize=16,  va='top', ha='right', weight='bold')


    le = [Line2D([0], [0], c='r',ls='--', label='fit'),
          Line2D([0], [0], c='k', label='PSTH')]

    axs['Ea'].legend(handles=le,
     loc='lower right', ncol=1).set_draggable(True)

    n, bins, patches = axs['F'].hist(x=r2s, bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)

    axs['F'].set_xlabel(r'$r^2$')
    axs['F'].set_ylabel('number of neurons')
    axs['F'].set_title('goodness of PSTH \n reconstruction')
    axs['F'].text(-0.1, 1.15, 'F', transform=axs['F'].transAxes,
      fontsize=16,  va='top', ha='right', weight='bold')

    # per region dim red
    ms = ['G','H','I','J','K']
    ms2 = ['Ga','Ha','Ia','Ja','Ka']

    k = 0

    p_r = {}
    for reg in D:
        if reg is None:
            continue

        y2 = y[regs == reg]
        labs2 = labs[regs == reg]

#        pca = PCA(n_components=2)
#        pca.fit(y2)
#        emb = pca.transform(y2)

        emb2 = emb[regs == reg]

        cols_lab = [lab_cols[b[x]] for x in labs2]
        axs[ms[k]].scatter(emb2[:,0],emb2[:,1],marker='o',c=cols_lab, s=2)

        labs_ = Counter(labs2)

        cents = {}
        for lab in labs_:
            cents[lab] = np.mean(emb2[labs2 == lab],axis=0)

        # plot means
        for lab in cents:
            axs[ms[k]].scatter(cents[lab][0],cents[lab][1],
                          s=800,marker='x',color=lab_cols[b[lab]])

        # plot confidence ellipses
        for lab in labs_:
            x = emb2[labs2 == lab]
            confidence_ellipse(x[:,0], x[:,1], axs[ms[k]], n_std=1.0,
                               edgecolor=lab_cols[b[lab]])

        # shuffle test
        nrand = 20  #random lab allocations
        centsr = []
        for shuf in range(nrand):
            labsr = labs2.copy()
            random.shuffle(labsr)
            cenr = {}
            for lab in labs_:
                cenr[lab] = np.mean(emb2[labsr == lab],axis=0)
            centsr.append(cenr)


        ps = {}
        for lab in cents:
            cs = np.mean([cents[l] for l in cents if l!=lab],axis=0)
            dist = distE(cents[lab],cs)

            null_d = [distE(cenr[lab],
                      np.mean([cenr[l] for l in cenr if l!=lab],axis=0))
                      for cenr in centsr]
            p = 1 - (0.01 * percentileofscore(null_d,dist))
            ps[lab] = np.round(p,3)
        p_r[reg] = ps

#        comb = combinations(cents, 2)

#        ps = {}
#        for pair in comb:
#            dist = distE(cents[pair[0]],cents[pair[1]])
#            null_d = [distE(cenr[pair[0]],cenr[pair[1]]) for cenr in centsr]
#            p = 1 - (0.01 * percentileofscore(null_d,dist))
#            ps[pair] = np.round(p,3)
#        p_r[reg] = ps

        axs[ms[k]].set_title(reg)
        axs[ms[k]].set_xlabel('embedding dim 1')
        axs[ms[k]].set_ylabel('embedding dim 2')
        axs[ms[k]].text(-0.1, 1.15, ms[k], transform=axs[ms[k]].transAxes,
          fontsize=16,  va='top', ha='right', weight='bold')

        axs[ms[k]].sharex(axs['A'])
        axs[ms[k]].sharey(axs['A'])

        # plot average PSTHs across labs
        for lab in labs_:

            mes = np.mean((y2[labs2 == lab])/T_BIN,axis=0)
            ste = np.std((y2[labs2 == lab])/T_BIN,axis=0)/np.sqrt(len(y2[labs2 == lab]))


            axs[ms2[k]].plot(xs,mes,color=lab_cols[b[lab]])
            axs[ms2[k]].fill_between(xs, mes + ste, mes - ste,
                                  color=lab_cols[b[lab]],alpha=0.2)

        axs[ms2[k]].set_title(reg)
        axs[ms2[k]].set_xlabel('time [sec]')
        axs[ms2[k]].set_ylabel('firing rate [Hz]')
        for x in np.array([25, 100])*T_BIN:
            axs[ms2[k]].axvline(x=x, lw=0.5, linestyle='--',
                                   c='g',label='motion start')

        axs[ms2[k]].axvline(x=75*T_BIN, lw=2, linestyle='-',
                               c='cyan',label='trial cut')

        axs[ms2[k]].text(-0.1, 1.15, ms2[k], transform=axs[ms2[k]].transAxes,
          fontsize=16,  va='top', ha='right', weight='bold')

        le = [Line2D([0], [0], color='g', lw=0.5, ls='--', label='motion start'),
              Line2D([0], [0], color='cyan', lw=2, ls='-', label='trial cut')]

#        axs[ms2[k]].legend(handles=le,
#         loc='lower right', ncol=1).set_draggable(True)

        axs[ms2[k]].sharex(axs['C'])
        axs[ms2[k]].sharey(axs['C'])

        k+=1

    # plot permutation test p values for regional scatters
    a = np.zeros((len(p_r),len(p_r[list(p_r.keys())[0]])))

    # multiple comparison correction
    pvals = [p_r[reg][lab] for  reg in p_r for lab in p_r[reg]]
    _, pvals_c, _, _ = multipletests(pvals, 0.05, method='fdr_bh')

    p_rc = {}
    i = 0
    for reg in p_r:
        p_rc[reg] = {}
        for lab in p_r[reg]:
            p_rc[reg][lab] = pvals_c[i]
            i += 1

    i = 0
    for reg in p_r:
        j = 0
        for pair in p_r[reg]:
            #if p_r[reg][pair] < 0.05:
            a[i,j] = p_r[reg][pair]
            j += 1
        i += 1

    im = axs['L'].imshow(a, cmap='cool',aspect="auto",#'Greys'
        interpolation='none')
    cb = fig.colorbar(im, ax=axs['L'], location='right', anchor=(0, 0.3), shrink=0.7)
    cb.ax.set_ylabel('p')
    axs['L'].set_xticks(range(len(p_r[reg].keys())))
    axs['L'].set_xticklabels([b[lab] for lab in p_r[reg].keys()],rotation = 90)
    axs['L'].set_yticks(range(len(p_r.keys())))
    axs['L'].set_yticklabels(p_r.keys())

    axs['L'].set_title(f'permutation test')
    axs['L'].set_xlabel(f'labs')
    axs['L'].set_ylabel('regions')
    axs['L'].text(-0.1, 1.15, 'L', transform=axs['L'].transAxes,
      fontsize=16,  va='top', ha='right', weight='bold')

    plt.suptitle(f'{algo}; dim reduction of PSTH; {len(y)} clusters'
                 f'; responsive only {rm_unre}; align = {align}; split = {split}')
    plt.tight_layout()



def plot_reaction_time_hists(rts=None):

    fig = plt.figure()
    lab3 = ['wittenlab','zadorlab','cortexlab']
    if rts is None:
        inserts = get_repeated_sites()
        eids = np.array(inserts)[:,0]

        rts = []
        for eid in eids:
            lab = str(one.eid2path(eid)).split('/')[4]
            if lab in lab3:
                continue
            wheelMoves = one.load_object(eid, 'wheelMoves')
            A = wheelMoves['intervals'][:,0]
            trials = one.load_object(eid, 'trials')
            evts = ['stimOn_times', 'feedback_times', 'probabilityLeft',
                    'choice', 'feedbackType']

            for tr in range(len(trials['intervals'])):

                # skip trial if any key info is nan
                if any(np.isnan([trials[k][tr] for k in evts])):
                    continue

                # skip trial if duration is too long
                if trials['feedback_times'][tr] - trials['stimOn_times'][tr] > 10:
                    continue


                b = trials['stimOn_times'][tr]
                c = trials['feedback_times'][tr]
                # making sure the motion onset time is in a coupled interval
                ind = np.where((A > b) & (A < c), True, False)
                if all(~ind):
                    #print(f'non-still start, trial {tr} and eid {eid}')
                    continue
                a=A[ind][0]
                rts.append(a-b)

    rtsf = np.array(rts)

    plt.hist(rtsf[np.where(rtsf<5)[0]],bins=200)
    plt.xlabel('reaction time [sec]')
    plt.ylabel('frequency')
    plt.title('pooled reaction times \n of repeated site sessions')





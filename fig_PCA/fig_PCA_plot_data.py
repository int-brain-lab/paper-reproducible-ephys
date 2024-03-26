import numpy as np
from collections import Counter

from scipy.stats import ks_2samp
import random
from copy import deepcopy
import pandas as pd
from datetime import datetime

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import r2_score


from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import string
from matplotlib import cm
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.sankey import Sankey
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import matplotlib as mpl

from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from datetime import datetime

from reproducible_ephys_functions import (filter_recordings, 
save_figure_path, labs, figure_style)
from fig_PCA.fig_PCA_load_data import load_dataframe, load_data

import warnings
warnings.filterwarnings("ignore")


T_BIN = 0.02  # time bin size in seconds
_, b, lab_cols = labs()


# set figure style
figure_style()


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
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std  # /np.sqrt(len(x))
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std  # /np.sqrt(len(y))
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def ecdf(a):
    '''
    get cummulative distribution function
    '''
    x, counts = np.unique(a, return_counts=True)
    cusum = np.cumsum(counts)
    return x, cusum / cusum[-1]


def distE(x, y):
    # Euclidean distance of points for permutation test
    return np.sqrt(np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y))


def perm_test_shift(inclu=False, fdr=True, sig_lev=0.01, ax = None):

    '''
    do KS test with multiples of std of values shifted
    until target issignificantly different to remaining
    '''
    emb, regs, labss = all_panels(get_dat=True) 
    
    # abbreviate lab names
    labs0 = [b[x] for x in labss]
    labss = np.array(labs0)
       
    regs_ = list(Counter(regs))
    labs_ = list(Counter(labss))

    td = {'regs':regs,'labs':labss}

    res = []
    As = []
    # shift firs PCs by multiples of their std 
    n_stds = 1
    steps = 10
    shifts = np.linspace(0,n_stds,steps)
    
    for shift in shifts:
     
        RKS = {}

        for tarn in td:
            
            tar = td[tarn]

            for reg in regs_ + ['all']:
                if tarn != 'labs':
                    if reg != 'all':
                        continue
                    else:
                        emb0 = emb
                        tar0 = tar
                else:
                    if reg == 'all':
                        emb0 = emb
                        tar0 = tar
                                   
                    else:
                        emb0 = emb[regs == reg]
                        tar0 = tar[regs == reg]
                    
                tar_ = Counter(tar0)
                
                gs = {}  # first PCs per class (for KS test)
                gsi = {}  # first PCs per remaining classes
                
                for x in tar_:
                
                    # here the shift occurs
                    x0 = emb0[tar0 == x][:,0]
                    gs[x] = x0 + shift * np.std(x0)
                    gsi[x] = emb0[tar0 != x][:,0]                
               
                pKS = {}
                      
                for x in tar_:
                    if inclu:
                        g_ = emb0[:,0]
                    else:
                        g_ = gsi[x]
                    
                    _, pKS[x] = ks_2samp(g_, gs[x])
                    
                if fdr:
                    _, pls_cKS, _, _ = multipletests([pKS[x] for x in pKS],
                                                    sig_lev, method='fdr_bh')
                else:
                    pls_cKS = [pKS[x] for x in pKS]

                resKS = dict(zip([x for x in pKS], 
                                 [np.round(y,4) for y in pls_cKS]))
                RKS[tarn+'_'+reg] = resKS  

        res.append(RKS)

        # matrix to summarize p-values        
        AKS = np.empty([len(regs_) + 1, len(labs_) + 1])
        AKS[:] = np.nan


        # fill entries for region contsraint tests    
        for i in range(len(regs_)):
            for j in range(len(labs_)):
                if labs_[j] not in RKS[f'labs_{regs_[i]}']:
                    continue        
                else:
                    AKS[i,j] = RKS[f'labs_{regs_[i]}'][labs_[j]]

        # add for tests that use all cells
        j = 0
        for lab in RKS['labs_all']:        
            AKS[-1,j] = RKS['labs_all'][lab]
            j += 1 
            
        i = 0
        for reg in RKS['regs_all']:       
            AKS[i,-1] = RKS['regs_all'][reg]
            i += 1
            
        As.append(AKS)  

    A = np.array(As)
    
    _, rs, cols = A.shape
    
    a0 = np.empty((rs,cols))
    a0[:] = np.nan
    
    for i in range(rs):
        for j in range(cols):
            if any(np.isnan(A[:,i,j])):
                continue    
            a0[i,j] = np.where(A[:,i,j]<sig_lev)[0][0]


    '''
    plotting
    '''

    if ax is None:
        fig, ax = plt.subplots(figsize=(3,3))
    else:
        fig = plt.gcf()


    ncols = int(np.nanmax(a0)) + 1
    cmap = mpl.colors.ListedColormap(plt.cm.Blues(np.linspace(0.3, 1, ncols)))
    #cmap.set_under((.8, .8, .8, 1.0))

    # continue here
    sns.heatmap(a0.T, cmap=cmap, square=True,
                cbar=True,
                annot=False, annot_kws={"size": 5},
                linewidths=.5, fmt='.2f', 
                ax=ax)                
    
    cbar = ax.collections[0].colorbar
    cbar.set_ticks(np.linspace(1, ncols, 2 * ncols + 1)[1::2] -1)
    cbar.set_ticklabels(np.arange(0, ncols)/10)
    cbar.set_label('shift until sig. [std]')
    
    
    ax.set(xlabel='', ylabel='', 
              xticks=np.arange(len(regs_ + ['all'])) + 0.5, 
              yticks=np.arange(len(labs_ + ['all'])) + 0.5,
              title='power analysis')
              
    ax.set_yticklabels(labs_ + ['all'], 
                          va='center', rotation=0)
    ax.set_xticklabels(regs_ + ['all'], rotation=90)
    
    # separate 'all' from others via lines
    ax.axvline(x=len(regs_),c='k', linewidth=2)
    ax.axhline(y=len(labs_),c='k', linewidth=2)

    fig = plt.gcf()
    fig.tight_layout()



def perm_test(inclu=False, print_=False, 
              nrand=2000, sig_lev =0.01, fdr = True, ax = None,
              plot_=True, samp=True):

    '''
    compute the distance of the mean of a subgroup
    to that of the remaining points (pack)
    
    nrand: random permutations
    emb: array of 2d points
    target: either 
    restr: restricted to an area (target must be lab)
    incl: include tested cells into pack
    samp: for region-targeted test, restrict to random subset 
    '''
        
    emb00, regs00, labss00 = all_panels(get_dat=True)
    labs_ = list(Counter(labss00))
    regs_ = Counter(regs00)
    
    assert len(emb00) == len(regs00) == len(labss00), 'mismatch' 
    
    
    if samp:
        

        fig_path = save_figure_path(figure='fig_PCA')
        
        if fig_path.joinpath('A.npy').is_file():
            print('loading sample averaged test results')
            AKS = np.load(fig_path.joinpath('A.npy'))

        else:
            print('computing 100 sample averages, KS only')
                        
            As = []
            for i in range(100):

                # restrict KS analysis to subset of cells
                reg0 = random.sample(list(regs_),1)[0]
                print('restricting tests on subsets of cells')
                n0 = dict(Counter(regs00))[reg0]
                print(f'sampling randomly {n0} cells ({reg0})')
                s0 = random.sample(range(len(regs00)),n0)
                
                emb = emb00[s0]
                regs = regs00[s0]
                labss = labss00[s0]
                
                td = {'regs':regs,'labs':labss}
                RKS = {}

                for tarn in td:
                    
                    tar = td[tarn]

                    for reg in list(regs_) + ['all']:
                        if tarn != 'labs':
                            if reg != 'all':
                                continue
                            else:
                                emb0 = emb
                                tar0 = tar
                        else:
                            if reg == 'all':
                                emb0 = emb
                                tar0 = tar
                                           
                            else:
                                emb0 = emb[regs == reg]
                                tar0 = tar[regs == reg]
                            
                        tar_ = Counter(tar0)
                        gs = {}  # first PCs per class (for KS test)
                        gsi = {}  # first PCs per remaining classes
                        
                        for x in tar_:
                            gs[x] = emb0[tar0 == x][:,0]
                            gsi[x] = emb0[tar0 != x][:,0]                

                        centsr = []  # null_d
                        centsir = []  # null_d inverse
                        pKS = {}
                              
                        for x in tar_:
                            if inclu:
                                g_ = emb0[:,0]
                            else:
                                g_ = gsi[x]
                            
                            _, pKS[x] = ks_2samp(g_, gs[x])
                                
                        if fdr:
                            _, pls_cKS, _, _ = multipletests(
                                                  [pKS[x] for x in pKS],
                                                  sig_lev, method='fdr_bh')
                        else:
                            pls_cKS = [pKS[x] for x in pKS]

                        resKS = dict(zip([x for x in pKS], 
                                         [np.round(y,4) for y in pls_cKS]))
                        RKS[tarn+'_'+reg] = resKS  

                        if print_:            
                            print('')     
                            print(f'target:{tarn}, region {reg},'
                                  f' inclu = {inclu}')
                            print('')
                            print(f'KS p-values')           
                            print([(x,resKS[x]) for x in resKS 
                                    if resKS[x] < sig_lev])               
                            print('')


                # matrix to summarize p-values    
                AKS = np.empty([len(regs_) + 1, len(labs_) + 1])
                AKS[:] = np.nan


                # fill entries for region contsraint tests    
                for i in range(len(regs_)):
                    for j in range(len(labs_)):
                        if labs_[j] not in RKS[f'labs_{list(regs_)[i]}']:
                            if print_:
                                print(labs_[j], 'not in',
                                f'labs_{list(regs_)[i]}')
                            continue        
                        else:
                            AKS[i,j] = RKS[
                                f'labs_{list(regs_)[i]}'][labs_[j]] + 0.0001


                # add for tests that use all cells
                j = 0
                for lab in RKS['labs_all']:        
                    AKS[-1,j] = RKS['labs_all'][lab] + 0.0001
                                    
                    j += 1

                i = 0
                for reg in RKS['regs_all']:       
                    AKS[i,-1] = RKS['regs_all'][reg] + 0.0001
                                                                                           
                    i += 1             
            
                As.append(AKS)
            A = np.nanmean(As,axis=0)
            np.save(fig_path.joinpath('A.npy'),A)
            AKS = A

    else:
        print('computing KS and dist test on all cells')
        
        emb = emb00
        regs = regs00
        labss = labss00 
        
        if print_:    
            print('data')
            print(emb.shape)
            print('labs')
            print(Counter(labss))
            print('regs')
            print(Counter(regs))
            print('')
            print(f'only occuraces of p < {sig_lev} are shown')
        
        if nrand < 1000: # random region allocations
            print('put nrand back to 1000')
            
        td = {'regs':regs,'labs':labss}

        R = {}
        RKS = {}

        for tarn in td:
            
            tar = td[tarn]

            for reg in list(regs_) + ['all']:
                if tarn != 'labs':
                    if reg != 'all':
                        continue
                    else:
                        emb0 = emb
                        tar0 = tar
                else:
                    if reg == 'all':
                        emb0 = emb
                        tar0 = tar
                                   
                    else:
                        emb0 = emb[regs == reg]
                        tar0 = tar[regs == reg]
                    
                tar_ = Counter(tar0)
                
                cents = {}  # centers per class (region or lab) 
                centsi = {}  # centers of remaining classes
                
                gs = {}  # first PCs per class (for KS test)
                gsi = {}  # first PCs per remaining classes
                
                for x in tar_:
                    cents[x] = np.mean(emb0[tar0 == x], axis=0)
                    centsi[x] = np.mean(emb0[tar0 != x], axis=0)
                    
                    gs[x] = emb0[tar0 == x][:,0]
                    gsi[x] = emb0[tar0 != x][:,0]                

                centsr = []  # null_d
                centsir = []  # null_d inverse
          
                for shuf in range(nrand):
                    tarr = tar0.copy()
                    random.shuffle(tarr)
                    cenr = {}
                    cenri = {}
                    
                    for x in tar_:
                        cenr[x] = np.mean(emb0[tarr == x], axis=0)
                        cenri[x] = np.mean(emb0[tarr != x], axis=0)
                                            
                    centsr.append(cenr)
                    centsir.append(cenri)            

                pr = {}
                pKS = {}
                
                      
                for x in tar_:
                    if inclu:
                        cs = np.mean(emb0, axis=0)
                        g_ = emb0[:,0]
                    else:
                        cs = centsi[x]
                        g_ = gsi[x]
                    
                    _, pKS[x] = ks_2samp(g_, gs[x])
                        
                    dist = distE(cents[x], cs)
                    null_d = [distE(cenr[x],cenri[x]) 
                              for cenr, cenri in zip(centsr,centsir)]
            
                    pr[x] = np.mean(np.array(null_d + [dist]) >= dist) 
                    
                if fdr:
                    _, pls_c, _, _ = multipletests([pr[x] for x in pr],
                                                    sig_lev, method='fdr_bh')
                    _, pls_cKS, _, _ = multipletests([pKS[x] for x in pKS],
                                                    sig_lev, method='fdr_bh')
                else:
                    pls_c = [pr[x] for x in pr]
                    pls_cKS = [pKS[x] for x in pKS]

                res = dict(zip([x for x in pr], 
                               [np.round(y,4) for y in pls_c]))
                               
                R[tarn+'_'+reg] = res

                resKS = dict(zip([x for x in pKS], 
                                 [np.round(y,4) for y in
                                  pls_cKS]))              
                RKS[tarn+'_'+reg] = resKS  

        
                if print_:            
                    print('')     
                    print(f'target:{tarn}, region {reg}, inclu = {inclu}')
                    print('') 
                    print(f'perm test 2d distance to pack, p-values')    
                    print([(x,res[x]) for x in res if res[x] < sig_lev])
                    print('')
                    print(f'KS p-values')           
                    print([(x,resKS[x]) for x in resKS 
                            if resKS[x] < sig_lev])               
                    print('')


        # matrix to summarize p-values A for dist test, AKS for KS  
        A = np.empty([len(regs_), len(labs_) + 1])
        A[:] = np.nan
        
        AKS = np.empty([len(regs_) + 1, len(labs_) + 1])
        AKS[:] = np.nan


        # fill entries for region contsraint tests    
        for i in range(len(regs_)):
            for j in range(len(labs_)):
                if labs_[j] not in RKS[f'labs_{list(regs_)[i]}']:
                    if print_:
                        print(labs_[j], 'not in', f'labs_{list(regs_)[i]}')
                    continue        
                else:
                    A[i,j] = R[f'labs_{list(regs_)[i]}'][labs_[j]]
                    AKS[i,j] = RKS[f'labs_{list(regs_)[i]}'][labs_[j]] + 0.0001

        # add for tests that use all cells
        j = 0
        for lab in RKS['labs_all']:        
            A[-1,j] = R['labs_all'][lab]
            AKS[-1,j] = RKS['labs_all'][lab] + 0.0001
                            
            j += 1
      
        i = 0
        for reg in RKS['regs_all']:       
            A[i,-1] = R['regs_all'][reg]   
            AKS[i,-1] = RKS['regs_all'][reg] + 0.0001
                                                                                   
            i += 1 

    # Create colormap
    RdYlGn = cm.get_cmap('RdYlGn', 256)(np.linspace(0, 1, 800))

    color_array = np.vstack([np.tile(
            np.concatenate((to_rgb('darkviolet'), 
            [1])), (200, 1)), RdYlGn])
    newcmp = ListedColormap(color_array)

    if ax is None:
        fig, ax = plt.subplots(figsize=(3,3))
    else:
        fig = plt.gcf()

    if not samp:     
        axin = inset_axes(ax, width="5%", height="80%", 
                          loc='lower right', borderpad=0,
                          bbox_to_anchor=(0.1, 0.1, 1, 1), 
                          bbox_transform=ax.transAxes)
                          
    # continue here (change AKS to A for dist test)
    sns.heatmap(np.log10(AKS.T), cmap=newcmp, square=True,
                cbar=True if not samp else False, 
                cbar_ax=axin if not samp else None,
                annot=False, annot_kws={"size": 5},
                linewidths=.5, fmt='.2f', vmin=-2.5, vmax=np.log10(1), 
                ax=ax)
    
    if not samp:            
        cbar = ax.collections[0].colorbar
        cbar.set_ticks(np.log10([0.01, 0.1, 1]))
        cbar.set_ticklabels([0.01, 0.1, 1])
        cbar.set_label('p-value (log scale)', labelpad=8)
    
    ax.set(xlabel='', ylabel='', 
              xticks=np.arange(len(list(regs_) + ['all'])) + 0.5, 
              yticks=np.arange(len([b[x] for x in labs_] + ['all'])) + 0.5,
              title='KS test mean' if samp else 'KS test')
    ax.set_yticklabels([b[x] for x in labs_] + ['all'], 
                          va='center', rotation=0)
    ax.set_xticklabels(list(regs_) + ['all'], rotation=90)
    
    # separate 'all' from others via lines
    ax.axvline(x=len(regs_),c='k', linewidth=2)
    ax.axhline(y=len(labs_),c='k', linewidth=2)

    fig = plt.gcf()
    fig.tight_layout()


def all_panels(rm_unre=True, align='move', split='rt', 
               xyz_res=False, re_rank=2, fdr=True, permute_include=True,
               nrand = 2000, sig_lev = 0.01, inclu = False, 
               perm_tests=True, get_dat=False, freeze='freeze_2024_03',
               get_info=False):
                             
    '''
    Plotting main figure and supp figure;
    See mosaic for panels
    incl = True --> include lab in pack for tests
    '''
               
    # load metainfo df, row per cell
    concat_df = load_dataframe()
    
    # load PSTHs, one per cell
    data = load_data(event=align, split=split, smoothing='none')
    all_frs = data['all_frs']

    ts = 'fast|slow RT PETH'
    
    # include has minimum number of clusters as being 3
    concat_df = filter_recordings(concat_df, min_regions=0)

    if permute_include:
        
        all_frs = all_frs[concat_df['permute_include'] == 1, :]
        concat_df = concat_df[concat_df['permute_include'] == 1].reset_index()

    if rm_unre:
        # restrict to responsive units
        all_frs = all_frs[concat_df['responsive']]
        concat_df = concat_df[concat_df["responsive"]].reset_index()

    # all having same length of #cells
    y = all_frs
    regs = concat_df['region'].values
    labs = concat_df['lab'].values
    
    print(len(Counter(concat_df['pid'].values)), 'insertions')
    
    xyz = np.array([concat_df[g].values for g in ['x', 'y', 'z']]).T

    # merge hofer and mirsicflogel labs
    labs[labs == 'mrsicflogellab'] = 'hoferlab'
    
    # PPC to VISa/am
    regs[regs == 'PPC'] = 'VISa/am'
    
    
    # PCA embedding
    pca = PCA(n_components=2)
    pca.fit(y)
    emb = pca.transform(y)

    if get_dat:
        return emb, regs, labs
  
    if get_info:
        return concat_df

    if xyz_res:
        clf = Ridge(alpha=0)
        clf.fit(xyz, emb)
        emb_p = clf.predict(xyz)
        print('xyz influence subtracted')
        print('r**2', r2_score(emb, emb_p))
        return

    # get reproduction using serveral first PCs
    u, s, vh = np.linalg.svd(y)
    y_res = {}

    for re_rank in [1, 2, 3]:
        S_star = np.zeros(y.shape)
        for i in range(re_rank):
            S_star[i, i] = s[i]
        y_res[re_rank] = (np.array(np.matrix(u) * 
                          np.matrix(S_star) * np.matrix(vh)))

    xs = np.arange(len(y[0])) * T_BIN
    xs = xs[:len(xs) // 2]  # for just plotting first PSTH

    # get a sessions idx per cell (for err bars later)
    cps = [k for k in range(1, len(labs)) if labs[k] != labs[k - 1]]
    k = 0
    sess = []
    cps.append(len(labs))  # just to avoid last idx
    for i in range(len(labs)):
        if i == cps[k]:
            k += 1
        sess.append(k)

    sess = np.array(sess)

    fig = plt.figure(figsize=(8, 7), facecolor='w')
    figs = plt.figure(figsize=(11, 10), facecolor='w')

    inner = [['Ea'],
             ['Eb']]

#    mosaic = [[inner, 'F', 'KS', 'KSmean'],
#              ['B','B', 'D', 'KSregs'],
#              ['c_labs', 'c_labs', 'm_labs', 'KSlabs']]


    mosaic = [[inner, 'F','B','B'],
              ['D', 'KSregs','c_labs', 'c_labs'],
              ['m_labs', 'KSlabs', 'KS', 'KSmean']]
    

    mosaic_supp = [['Ha', 'Hb', 'H'],
                   ['Ia', 'Ib', 'I'],
                   ['Ja', 'Jb', 'J'],
                   ['Ka', 'Kb', 'K'],
                   ['Ga', 'Gb', 'G']]
                   
    mf = [item for sublist in mosaic for item in sublist]               
    mf[0] = 'Ea'
    panel_n = dict(zip(list(Counter(mf)), string.ascii_lowercase))
    
    # custom swap panel labels:
    panel_n['KSmean'] = 'j'
    panel_n['KS'] = 'i'
    panel_n['B'] = 'c'
    panel_n['c_labs'] = 'f'
    panel_n['D'] = 'd'
    panel_n['m_labs'] = 'g' 
    panel_n['KSlabs'] = 'h' 
    panel_n['KSregs'] = 'e' 

    mfs = np.array(mosaic_supp, dtype=object).flatten()
    panel_ns = dict(zip(mfs, string.ascii_lowercase))

    axs = fig.subplot_mosaic(mosaic)
    axss = figs.subplot_mosaic(mosaic_supp)
    
    # despine all plots
    for key in axs:
        axs[key].spines['top'].set_visible(False)
        axs[key].spines['right'].set_visible(False)
    for key in axss:
        axss[key].spines['top'].set_visible(False)
        axss[key].spines['right'].set_visible(False)        
            
    labs_ = Counter(labs)

    le_labs = [Patch(facecolor=lab_cols[b[lab]], 
               edgecolor=lab_cols[b[lab]], label=b[lab]) for lab in labs_]
                 
  
    '''
    ###
    perm tests, plot
    ###
    '''

    if perm_tests:
        # run all permutation tests
        perm_test(inclu=inclu, 
                  nrand=nrand, sig_lev =sig_lev, 
                  fdr = fdr, ax=axs['KS'],samp=False) 
                  
        # average across subset sampling
        perm_test(inclu=inclu, 
                  nrand=nrand, sig_lev =sig_lev, 
                  fdr = fdr, ax=axs['KSmean'],samp=True)                   
                  
                  
#        # power analysis          
#        perm_test_shift(inclu=inclu,sig_lev =sig_lev,fdr = fdr, 
#                        ax=axs['KSshift'])
                        
        # put panel label
        for pan in ['KS', 'KSmean']:              
            axs[pan].text(-0.1, 1.3, panel_n[pan],
                            transform=axs[pan].transAxes,
                            fontsize=16, va='top',
                            ha='right', weight='bold')    

    '''
    ###
    plot scatter, all cells, colored by reg
    ###
    '''
    Dc = figure_style(return_colors=True)
    Dc['VISa/am'] = Dc['PPC']

    # scatter 2d PCs
    cols_reg = [Dc[x] for x in regs]
    axs['B'].scatter(emb[:, 0], emb[:, 1], marker='o', c=cols_reg, s=2)
    
    # centers per region 
    regs_ = Counter(regs)    
    cents = {reg: np.mean(emb[regs == reg], axis=0)
             for reg in regs_}   
    
    
#    for reg in cents:
#        # plot means
#        axs['B'].scatter(cents[reg][0], cents[reg][1], 
#                         s=500, marker='x', color=Dc[reg])
#                         
#        # plot confidence ellipses
#        x = emb[regs == reg]
#        confidence_ellipse(x[:, 0], x[:, 1], axs['B'], 
#                           n_std=1.0, edgecolor=Dc[reg])        

    le = [Patch(facecolor=Dc[reg], edgecolor=Dc[reg], 
                label=reg) for reg in regs_]
                       

    axs['B'].legend(handles=le, bbox_to_anchor=(0.3, 1), 
                    loc='lower left', ncol=3, frameon=False,
                    prop={'size': 7}).set_draggable(True)

    axs['B'].set_title('regions', loc='left')
    axs['B'].set_xlabel('embedding dim 1')
    axs['B'].set_ylabel('embedding dim 2')
    axs['B'].text(-0.1, 1.30, panel_n['B'], 
                  transform=axs['B'].transAxes, fontsize=16, 
                  va='top', ha='right', weight='bold')

    axs['B'].sharex(axss['H'])
    axs['B'].sharey(axss['H'])
    axs['D'].sharex(axs['m_labs'])
    axs['D'].sharey(axs['m_labs'])
    
    

    '''
    ###
    plot scatter, all cells, colored by labs
    ###
    '''
    
    centsl = {lab: np.mean(emb[labs == lab], axis=0)
              for lab in labs_}

    # scatter 2d PCs
    cols_lab = [lab_cols[b[x]] for x in labs]
    axs['c_labs'].scatter(emb[:, 0], emb[:, 1], marker='o', c=cols_lab, s=2)
        
    
#    for lab in centsl:
#        # plot means
#        axs['c_labs'].scatter(centsl[lab][0], centsl[lab][1], 
#                         s=500, marker='x', color=lab_cols[b[lab]])
#                         
#        # plot confidence ellipses
#        x = emb[labs == lab]
#        confidence_ellipse(x[:, 0], x[:, 1], axs['c_labs'], 
#                           n_std=1.0, edgecolor=lab_cols[b[lab]])        

    axs['c_labs'].legend(handles=le_labs, bbox_to_anchor=(0.3, 1), 
                    loc='lower left', ncol=3, frameon=False,
                    prop={'size': 7}).set_draggable(True)

    axs['c_labs'].set_title('labs', loc='left')
    axs['c_labs'].set_xlabel('embedding dim 1')
    axs['c_labs'].set_ylabel('embedding dim 2')
    axs['c_labs'].text(-0.1, 1.30, panel_n['c_labs'], 
                  transform=axs['c_labs'].transAxes, fontsize=16, 
                  va='top', ha='right', weight='bold')

    axs['c_labs'].sharex(axs['B'])
    axs['c_labs'].sharey(axs['B'])
    axs['D'].sharex(axs['m_labs'])
    axs['D'].sharey(axs['m_labs'])


    '''
    ###
    # plot average PSTHs across regions, all cells
    ###
    '''

    for reg in regs_:

        labsinreg = len(Counter(labs[regs == reg]))
        sessinreg = len(Counter(sess[regs == reg]))

        ms = np.mean((y[regs == reg]) / T_BIN, axis=0)
        ste = np.std((y[regs == reg]) / T_BIN, axis=0) / np.sqrt(sessinreg)

        # only plot one PSTH
        ms = ms[:len(xs)]
        ste = ste[:len(xs)]

        axs['D'].plot(xs - 0.5, ms, color=Dc[reg])
        axs['D'].fill_between(xs - 0.5, ms + ste, ms - ste, 
                              color=Dc[reg], alpha=0.2)

    for x in np.array([25]) * T_BIN:
        axs['D'].axvline(x=0, linewidth=0.5, linestyle='--', 
                         c='g', label='movement onset')

    axs['D'].set_xlabel('time from \n movement onset (s)')
    axs['D'].set_ylabel('Firing rate \n (spikes/s)')
    axs['D'].text(-0.1, 1.30, panel_n['D'], 
                  transform=axs['D'].transAxes, fontsize=16, 
                  va='top', ha='right', weight='bold')

    '''
    ###
    # plot average PSTHs arcoss labs, all cells
    ###
    '''

    for lab in labs_:

        labsinlab = len(Counter(labs[labs == lab]))
        sessinlab = len(Counter(sess[labs == lab]))

        ms = np.mean((y[labs == lab]) / T_BIN, axis=0)
        ste = np.std((y[labs == lab]) / T_BIN, axis=0) / np.sqrt(sessinlab)

        # only plot one PSTH
        ms = ms[:len(xs)]
        ste = ste[:len(xs)]

        axs['m_labs'].plot(xs - 0.5, ms, color=lab_cols[b[lab]])
        axs['m_labs'].fill_between(xs - 0.5, ms + ste, ms - ste, 
                              color=lab_cols[b[lab]], alpha=0.2)

    for x in np.array([25]) * T_BIN:
        axs['m_labs'].axvline(x=0, linewidth=0.5, linestyle='--', 
                         c='g', label='movement onset')

    axs['m_labs'].set_xlabel('time from \n movement onset (s)')
    axs['m_labs'].set_ylabel('Firing rate \n (spikes/s)')
    axs['m_labs'].text(-0.1, 1.30, panel_n['m_labs'], 
                  transform=axs['m_labs'].transAxes, fontsize=16, 
                  va='top', ha='right', weight='bold')


    '''
    example cell reconstruction
    '''

    le = [Line2D([0], [0], color='g', lw=0.5, ls='--', 
                 label='movement onset')]

    # illustrate example cells
    # 1 with great, 1 with bad reconstruction
    r2s_ = {}
    for re_rank in y_res:
        r2s = []
        iex = []
        for i in range(len(y)):
            r2s.append(r2_score(y[i], y_res[re_rank][i]))
            iex.append(i)

        r2s_[re_rank] = r2s
        if re_rank == 2:
            t = np.argsort(r2s)
            t_ = np.concatenate([t[169:170], t[-4:-3]])
            idxs = [iex[j] for j in t_]

    # split good and bad example cell into two panels
    ms = ['Ea', 'Eb']

    for k in range(len(idxs)):


        axs[ms[k]].plot(xs - 0.5, y[idxs[k]][:len(xs)] / T_BIN,
                        c='k', label='PETH')
        axs[ms[k]].plot(xs - 0.5, y_res[2][idxs[k]][:len(xs)] / T_BIN,
                        c='r', ls='--', label='2-PC-fit')

        for x in np.array([25]) * T_BIN:
            axs[ms[k]].axvline(x=0, linewidth=0.5, linestyle='--',
                               c='g', label='movement onset')

        axs[ms[k]].set_ylabel('Firing rate \n (spikes/s)')
        stext = rf'$R^2$={np.round(r2_score(y[idxs[k]], y_res[2][idxs[k]]), 2)}'
        axs[ms[k]].text(0.3, 1, stext, transform=axs[ms[k]].transAxes,
                        fontsize=7, va='top', ha='right')

        if k == 1:
            axs[ms[k]].set_xlabel('time from \n movement onset (s)')

        if k == 0:
            axs[ms[k]].text(-0.1, 1.6, panel_n[ms[k]],
                            transform=axs[ms[k]].transAxes,
                            fontsize=16, va='top',
                            ha='right', weight='bold')
                              
        k += 1

    for re_rank in r2s_:
        _, patches, _ = axs['F'].hist(x=r2s_[re_rank], 
                                      bins='auto', label=re_rank, 
                                      alpha=0.7, rwidth=0.85, 
                                      histtype=u'step', lw=2)

    leg = axs['F'].legend(ncol=1,frameon=False,
                          prop={'size': 7}, 
                          title="# PCs").set_draggable(True)
                          
    axs['F'].set_xlabel(r'$r^2$')
    axs['F'].set_ylabel('number of neurons')
    axs['F'].text(-0.1, 1.30, panel_n['F'], 
                  transform=axs['F'].transAxes, fontsize=16, 
                  va='top', ha='right', weight='bold')


    '''
    ####
    plot cummulative first PC dists and KS result inset
    for all cells region-targeted and 
    for all cells lab-targeted
    ###
    '''
    
    # lab-targeted    
    g_all = emb[:,0]
    gsi = {}
    g_labs = {}
    for lab in labs_:
        g_labs[lab] = emb[labs == lab][:,0]
        gsi[lab] = emb[labs != lab][:,0]
    
    ksr = {}                          
    for lab in labs_:
        x0, x1 = ecdf(g_labs[lab])
        ks,p = ks_2samp(g_all if inclu else gsi[lab], g_labs[lab])
        ksr[lab] = [ks,p]
        axs['KSlabs'].plot(x0, x1, color=lab_cols[b[lab]], 
                   drawstyle='steps-post', lw=1)
    
    # per reg multiple comparison corr                
    pvals_ = [ksr[lab][1] for lab in ksr]
    if fdr:
        _, pvals_c_, _, _ = multipletests(pvals_, sig_lev, method='fdr_bh')
    else:
        pvals_c_ = pvals_
    
    kk = 0
    
    for lab in ksr:
        ksr[lab][1] = pvals_c_[kk]
        kk += 1

    #axs['KSlabs'].set_title(reg, loc='left')
    axs['KSlabs'].set_xlabel('x')
    axs['KSlabs'].set_ylabel('P(PC1 < x)')
    axs['KSlabs'].text(-0.1, 1.30, panel_n['KSlabs'],
                     transform=axs['KSlabs'].transAxes,
                     fontsize=16, va='top',
                     ha='right', weight='bold')            

    # plot ks scores as bar plot inset with asterics for small p 
    axsir = inset_axes(axs['KSlabs'], width="30%", height="35%", 
                           borderpad=0, loc=4,
                           bbox_to_anchor=(-0.01,0.2,1,1), 
                           bbox_transform=axs['KSlabs'].transAxes)
               
    bars = axsir.bar(range(len(ksr)), [ksr[lab][0] for lab in ksr], 
                  color = [lab_cols[b[lab]] for lab in ksr])
                  
    axsir.set_xlabel('labs')
    axsir.set_xticks([])
    axsir.set_xticklabels([])
    axsir.set_ylabel('KS')
    axsir.spines['top'].set_visible(False)
    axsir.spines['right'].set_visible(False)
                 
    # put * on bars that are significant
    ii = 0
    for lab in ksr:
        ba = bars.patches[ii]
        if ksr[lab][1] < sig_lev:             
            axsir.annotate('*',
                               (ba.get_x() + ba.get_width() / 2,
                                ba.get_height()), ha='center', 
                                va='center', size=16, 
                                xytext=(0, 1),
                                textcoords='offset points')
        ii += 1

    # region-targeted    
    g_all = emb[:,0]
    gsi = {}
    g_regs = {}
    for reg in regs_:
        g_regs[reg] = emb[regs == reg][:,0]
        gsi[reg] = emb[regs != reg][:,0]
    
    ksr = {}                          
    for reg in regs_:
        x0, x1 = ecdf(g_regs[reg])
        ks,p = ks_2samp(g_all if inclu else gsi[reg], g_regs[reg])
        ksr[reg] = [ks,p]
        axs['KSregs'].plot(x0, x1, color=Dc[reg], 
                   drawstyle='steps-post', lw=1)
    
    # per reg multiple comparison corr                
    pvals_ = [ksr[reg][1] for reg in ksr]
    if fdr:
        _, pvals_c_, _, _ = multipletests(pvals_, sig_lev, method='fdr_bh')
    else:
        pvals_c_ = pvals_
    
    kk = 0
    
    for reg in ksr:
        ksr[reg][1] = pvals_c_[kk]
        kk += 1

    axs['KSregs'].set_xlabel('x')
    axs['KSregs'].set_ylabel('P(PC1 < x)')
    axs['KSregs'].text(-0.1, 1.30, panel_n['KSregs'],
                     transform=axs['KSregs'].transAxes,
                     fontsize=16, va='top',
                     ha='right', weight='bold')            

    # plot ks scores as bar plot inset with asterics for small p 
    axsir = inset_axes(axs['KSregs'], width="30%", height="35%", 
                           loc=4, borderpad=0,
                           bbox_to_anchor=(-0.01,0.2,1,1), 
                           bbox_transform=axs['KSregs'].transAxes)
               
    bars = axsir.bar(range(len(ksr)), [ksr[reg][0] for reg in ksr], 
                  color = [Dc[reg] for reg in ksr])
                  
    axsir.set_xlabel('regs')
    axsir.set_xticks([])
    axsir.set_xticklabels([])
    axsir.set_ylabel('KS')
    axsir.spines['top'].set_visible(False)
    axsir.spines['right'].set_visible(False)
                 
    # put * on bars that are significant
    ii = 0
    for reg in ksr:
        ba = bars.patches[ii]
        if ksr[reg][1] < sig_lev:             
            axsir.annotate('*',
                               (ba.get_x() + ba.get_width() / 2,
                                ba.get_height()), ha='center', 
                                va='center', size=16, 
                                xytext=(0, 1),
                                textcoords='offset points')
        ii += 1
 
 
    axs['KSregs'].sharex(axs['KSlabs'])
    axs['KSregs'].sharey(axs['KSlabs'])


    '''
    analysis per region
    '''
    
    ms = ['H', 'I', 'J', 'K','G']  # 2PCs scatter 'G',
    ms2 = ['Ha', 'Ia', 'Ja', 'Ka','Ga']  #'Ga',  average PETH
    ms3 = ['Hb', 'Ib', 'Jb', 'Kb','Gb']  #'Gb',  CDFs

    k = 0
    p_ = {}  # Guido's permutation test score
    p_ks = {}  # KS test on 1st PC
    D = ['VISa/am', 'CA1', 'DG', 'LP', 'PO']#'CA1', 

    
    axsi = []  # inset axes
    for reg in D:

        if reg is None:
            continue

        axs3 = axss
        panel_n3 = panel_ns

        y2 = y[regs == reg]
        labs2 = labs[regs == reg]
        sess2 = sess[regs == reg]
        emb2 = emb[regs == reg]
        
        cols_lab = [lab_cols[b[x]] for x in labs2]
        axs3[ms[k]].scatter(emb2[:, 0], emb2[:, 1], marker='o', 
                            c=cols_lab, s=2)

        labs_ = Counter(labs2)


        cents = {}
        for lab in labs_:
            cents[lab] = np.mean(emb2[labs2 == lab], axis=0)
            
#        # plot means
#        for lab in cents:
#            axs3[ms[k]].scatter(cents[lab][0], cents[lab][1], 
#                                s=500, marker='x', color=lab_cols[b[lab]])

#        # plot confidence ellipses
#        for lab in labs_:
#            x = emb2[labs2 == lab]
#            confidence_ellipse(x[:, 0], x[:, 1], axs3[ms[k]], 
#                               n_std=1.0, edgecolor=lab_cols[b[lab]])

        '''
        KS test using 1PCs
        '''
         
        g_all = emb2[:,0]
        gsi = {}
        g_labs = {}
        for lab in labs_:
            g_labs[lab] = emb2[labs2 == lab][:,0]
            gsi[lab] = emb2[labs2 != lab][:,0]
        
#        # plot CDFs of first PCs distributions and KS metrics
#        x0, x1 = ecdf(g_all)
#        axs3[ms3[k]].plot(x0, x1,
#                       label=f'all cells', lw=2,
#                       color = 'k', drawstyle='steps-post')
        ksr = {}                          
        for lab in labs_:
            x0, x1 = ecdf(g_labs[lab])
            ks,p = ks_2samp(g_all if inclu else gsi[lab], g_labs[lab])
            ksr[lab] = [ks,p]
            axs3[ms3[k]].plot(x0, x1, color=lab_cols[b[lab]], 
                       drawstyle='steps-post', lw=1)
        
        # per reg multiple comparison corr                
        pvals_ = [ksr[lab][1] for lab in ksr]
        if fdr:
            _, pvals_c_, _, _ = multipletests(pvals_, sig_lev, method='fdr_bh')
        else:
            pvals_c_ = pvals_
        
        kk = 0
        
        for lab in ksr:
            ksr[lab][1] = pvals_c_[kk]
            kk += 1
#            if ksr[lab][1] < sig_lev:
#                print(reg, lab, ksr[lab][0], ksr[lab][1])
        

        axs3[ms3[k]].set_title(reg, loc='left')
        axs3[ms3[k]].set_xlabel('x')
        axs3[ms3[k]].set_ylabel('P(PC1 < x)')
        axs3[ms3[k]].text(-0.1, 1.30, panel_n3[ms3[k]],
                         transform=axs3[ms3[k]].transAxes,
                         fontsize=16, va='top',
                         ha='right', weight='bold')            
        
        if k == 1:
            axs3[ms3[k]].legend(frameon=False, 
                                loc='upper left').set_draggable(True)

        # plot ks scores as bar plot inset with asterics for small p 
        axsi.append(inset_axes(axs3[ms3[k]], width="30%", height="35%", 
                               loc=4, borderpad=1,
                               bbox_to_anchor=(-0.02,0.1,1,1), 
                               bbox_transform=axs3[ms3[k]].transAxes))
                   
        bars = axsi[k].bar(range(len(ksr)), [ksr[lab][0] for lab in ksr], 
                      color = [lab_cols[b[lab]] for lab in ksr])
                      
        axsi[k].set_xlabel('labs')
        axsi[k].set_xticks([])
        axsi[k].set_xticklabels([])
        axsi[k].set_ylabel('KS')
        axsi[k].spines['top'].set_visible(False)
        axsi[k].spines['right'].set_visible(False)
                     
        # put * on bars that are significant
        ii = 0
        for lab in ksr:
            ba = bars.patches[ii]
            if ksr[lab][1] < sig_lev:             
                axsi[k].annotate('*',
                                   (ba.get_x() + ba.get_width() / 2,
                                    ba.get_height()), ha='center', 
                                    va='center', size=16, 
                                    xytext=(0, 1),
                                    textcoords='offset points')
            ii += 1
        
        '''
        ####
        '''
                
        axs3[ms[k]].set_title(reg, loc='left')
        axs3[ms[k]].set_xlabel('embedding dim 1 (PC1)' if k > 0 else
                               'embedding dim 1' )
        axs3[ms[k]].set_ylabel('embedding dim 2')
        axs3[ms[k]].text(-0.1, 1.30, panel_n3[ms[k]],
                         transform=axs3[ms[k]].transAxes,
                         fontsize=16, va='top',
                         ha='right', weight='bold')

        if ms2[k] == 'Ha':
            axs3[ms2[k]].legend(handles=le_labs, loc='lower left', 
                               bbox_to_anchor=(0.1, 1), ncol=3,frameon=False,
                               prop={'size': 8}).set_draggable(True)

        # plot average PSTHs across labs
        for lab in labs_:

            mes = np.mean((y2[labs2 == lab]) / T_BIN, axis=0)
            # normalise by the number of sessions, not the number of cells

            sessinlab = len(Counter(sess2[labs2 == lab]))

            ste = np.std((y2[labs2 == lab]) / T_BIN, axis=0) / np.sqrt(sessinlab)
            mes = mes[:len(xs)]
            ste = ste[:len(xs)]

            axs3[ms2[k]].plot(xs - 0.5, mes, color=lab_cols[b[lab]])
            axs3[ms2[k]].fill_between(xs - 0.5, mes + ste, mes - ste,
                                      color=lab_cols[b[lab]], alpha=0.2)

        axs3[ms2[k]].set_title(reg, loc='left')
        axs3[ms2[k]].set_xlabel('time from \n movement onset (s)')
        axs3[ms2[k]].set_ylabel('Firing rate \n (spikes/s)')
        
        for x in np.array([25]) * T_BIN:
            axs3[ms2[k]].axvline(x=0, lw=0.5, linestyle='--', 
                                 c='g', label='movement onset')

        axs3[ms2[k]].text(-0.1, 1.30, panel_n3[ms2[k]],
                          transform=axs3[ms2[k]].transAxes, 
                          fontsize=16, va='top', ha='right', weight='bold')

        le = [Line2D([0], [0], color='g', lw=0.5, ls='--', 
              label='movement onset')]

        axs3[ms[k]].sharex(axs['B'])
        axs3[ms[k]].sharey(axs['B'])
        axs3[ms2[k]].sharex(axs['D'])
        axs3[ms2[k]].sharey(axs['D'])

        k += 1


    axs['B'].set_xlim([-2, 1.5])
    axs['B'].set_ylim([-1.1, 1.1])
    axs['D'].set_xlim([-0.2, 0.8])
    axs['D'].set_ylim([-7, 12])

    fig.tight_layout()
    figs.tight_layout()

    # shift x position of KSshift heatmap to correct for long text
    B = axs['KS'].get_position()
    axs['KS'].set_position([B.x0 - 0.02, B.y0, B.width, B.height])

    fig_path = save_figure_path(figure='fig_PCA')
    print(f'Saving figures to {fig_path}')
    fig.savefig(fig_path.joinpath('figure_PCA.pdf'))
    figs.savefig(fig_path.joinpath('figure_PCA_supp1.pdf'))
        
    

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

import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from datetime import datetime
from scipy.stats import combine_pvalues

from reproducible_ephys_functions import (filter_recordings, 
save_figure_path, LAB_MAP, figure_style, get_row_coord, get_label_pos, BRAIN_REGIONS)
from fig_PCA.fig_PCA_load_data import load_dataframe, load_data
import figrid as fg

import warnings
warnings.filterwarnings("ignore")

PRINT_INFO = False


T_BIN = 0.02  # time bin size in seconds
_, b, lab_cols = LAB_MAP()

# canonical lab order
canon = ['danlab', 'mainenlab','churchlandlab',
        'angelakilab','wittenlab', 'hoferlab',
        'cortexlab', 'churchlandlab_ucla', 'steinmetzlab',
        'zadorlab']
        
canon_regs = [ "VISa/am",
    "CA1",
    "DG",
    "LP",
    "PO"]
       

# set figure style
figure_style()
Dc = figure_style(return_colors=True)
Dc['VISa/am'] = Dc['PPC']

# for significance testing
sig_lev=0.05

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


def perm_test(inclu=False, print_=False, rerun = True, align='move',
              nrand=2000, fdr = False, ax = None,
              plot_=True, samp=True, nns = 1000,):

    '''
    compute the distance of the mean of a subgroup
    to that of the remaining points (pack)
    
    nrand: random permutations
    emb: array of 2d points
    target: either 
    restr: restricted to an area (target must be lab)
    incl: include tested cells into pack
    samp: for region-targeted test, restrict to random subset 
    nns: how many samplings
    '''
        
    emb00, regs00, labss00, eids00 = all_panels(get_dat=True, align=align)
    labs__ = list(Counter(labss00))

    #order labs canonically
    labs_ = []

    for lab in canon:
        if lab in labs__:
            labs_.append(lab)    
    
    
    regs__ = Counter(regs00)
    regs_ = []
    for reg in canon_regs:
        if reg in regs__:
            regs_.append(reg)     
    
    
    assert len(emb00) == len(regs00) == len(labss00), 'mismatch' 
    
    
    if samp:
        

        fig_path = save_figure_path(figure='fig_PCA')
        
        if fig_path.joinpath('A.npy').is_file() and not rerun:
            print('loading sample averaged test results')
            AKS = np.load(fig_path.joinpath('A.npy'))

        else:
            print(f'computing {nns} sample averages, KS only')
                        
            As = []

            
            df00 = pd.DataFrame({
                'pc1': emb00[:,0],
                'pc2': emb00[:,1],
                'regs': regs00,
                'labs': labss00,
                'eids': eids00})

            for i in range(nns):
                
                
                RKS = {}

                for tarn in ['regs', 'labs']:

                    # KS on min subset of cells

                    min_cells_per_tarn = df00[tarn].value_counts().min()

                    sampled_df = (df00.groupby(tarn).apply(
                        lambda x: x.sample(n=min_cells_per_tarn, 
                        random_state=random.randint(1, 10000))
                        ).reset_index(drop=True))

                    emb = sampled_df['pc1'].values
                    regs = sampled_df['regs'].values
                    labss = sampled_df['labs'].values

                    td = {'regs':regs,'labs':labss}

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
                            gs[x] = emb0[tar0 == x]
                            gsi[x] = emb0[tar0 != x]               

                        centsr = []  # null_d
                        centsir = []  # null_d inverse
                        pKS = {}
                              
                        for x in tar_:
                            if inclu:
                                g_ = emb0
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
                
            #A = np.nanmean(As,axis=0)
            
            aa, bb = As[0].shape
            combined_p_values_matrix = np.empty((aa,bb))
                        
            num_matrices = len(As)  

            # Iterate over each element in the (6, 11) matrix
            for i in range(aa):
                for j in range(bb):
                    # Extract the p-value at position (i, j) from each matrix
                    p_values = [As[k][i, j] for 
                        k in range(num_matrices) 
                        if not np.isnan(As[k][i, j])]

                    # Check if there are any non-NaN p-values to process
                    if p_values:
                        # Combine p-values using Fisher's method
                        _, combined_p = combine_pvalues(p_values,
                             method='fisher')
                        combined_p_values_matrix[i, j] = combined_p
            
            A = combined_p_values_matrix
            np.save(fig_path.joinpath('A.npy'),A)

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

    if samp:
        axin = inset_axes(ax, width="10%", height="80%",
                          loc='lower right', borderpad=0,
                          bbox_to_anchor=(0.1, 0.1, 1, 1), 
                          bbox_transform=ax.transAxes)
                          
    sns.heatmap(np.log10(AKS.T), cmap=newcmp, square=True,
                cbar=True if samp else False,
                cbar_ax=axin if samp else None,
                annot=False, annot_kws={"size": 5},
                linewidths=.5, fmt='.2f', vmin=-2.5, vmax=np.log10(1), 
                ax=ax)
    
    if samp:
        cbar = ax.collections[0].colorbar
        cbar.set_ticks(np.log10([0.01, 0.1, 1]))
        cbar.set_ticklabels([0.01, 0.1, 1])
        cbar.set_label('p-value (log scale)', labelpad=2)
    
    ax.set(xlabel='', ylabel='', 
              xticks=np.arange(len(list(regs_) + ['all'])) + 0.5, 
              yticks=np.arange(len([b[x] for x in labs_] + ['all'])) + 0.5,
              title='KS test mean' if samp else 'KS test')
    ax.set_yticklabels([b[x] for x in labs_] + ['all'], 
                          va='center', rotation=0)
    ax.set_xticklabels(list(regs_) + ['all'], rotation=45, ha='right', rotation_mode="anchor")

    # separate 'all' from others via lines
    ax.axvline(x=len(regs_),c='k', linewidth=2)
    ax.axhline(y=len(labs_),c='k', linewidth=2)

    # fig = plt.gcf()
    # fig.tight_layout()


def all_panels(rm_unre=True, align='move', split='rt',
               xyz_res=False, fdr=False, permute_include=True,
               nrand = 2000, inclu = False, 
               perm_tests=True, get_dat=False, freeze='freeze_2024_03',
               get_info=False, nns = 1000, rerun=True):
                             
    '''
    Plotting main figure and supp figure;
    See mosaic for panels
    incl = True --> include lab in pack for tests
    '''

    # panel letter fontsize
    plfs = 10

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

    if PRINT_INFO:
        print('Figure 5')
        print(f'N_inst: {concat_df.institute.nunique()}, N_sess: {concat_df.eid.nunique()}, '
              f'N_mice: {concat_df.subject.nunique()}, N_cells: {len(concat_df)}')
        df_gb = concat_df.groupby('region')
        for reg in BRAIN_REGIONS:
            df_reg = df_gb.get_group(reg)
            print(f'Figure 5 supp 1: {reg}')
            print(f'N_inst: {df_reg.institute.nunique()}, N_sess: {df_reg.eid.nunique()}, '
                  f'N_mice: {df_reg.subject.nunique()}, N_cells: {len(df_reg)}')


    # all having same length of #cells
    y = all_frs
    regs = concat_df['region'].values
    labs = concat_df['lab'].values
    eids = concat_df['eid'].values
    
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
        return emb, regs, labs, eids
  
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

    xs = data['time']

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

    # Main figure
    width_m = 7
    height_m = 5.5
    figure_style()
    fig = plt.figure(figsize=(width_m, height_m), facecolor='w')

    padx = 0.5
    pady = 0.3
    xspans = get_row_coord(width_m, [1, 1, 2], hspace=0.8, pad=padx)
    yspans = get_row_coord(height_m, [1, 1, 1], hspace=1, pad=pady)
    yspan_inset = get_row_coord(height_m, [1, 1], pad=0, hspace=0.15, span=yspans[0])
    xspan_inset = get_row_coord(width_m, [10, 1], pad=0, hspace=0, span=xspans[2])
    xspans_row3 = get_row_coord(width_m, [1, 1], hspace=0.8, pad=padx)
    xspans_row3_1 = get_row_coord(width_m, [1, 1], hspace=0.8, pad=0, span=xspans_row3[0])
    xspans_row3_2 = get_row_coord(width_m, [1, 1], hspace=-0.4, pad=0.6, span=(xspans_row3[1][0], 1))
    print(xspans_row3)

    axs = {
        'Ea': fg.place_axes_on_grid(fig, xspan=xspans[0], yspan=yspan_inset[0]),  # A
        'Eb': fg.place_axes_on_grid(fig, xspan=xspans[0], yspan=yspan_inset[1]),
        'F': fg.place_axes_on_grid(fig, xspan=xspans[1], yspan=yspans[0]),  # B
        'B': fg.place_axes_on_grid(fig, xspan=xspan_inset[0], yspan=yspans[0]),  # C
        'C_2': fg.place_axes_on_grid(fig, xspan=xspan_inset[1], yspan=yspans[0]),
        'D': fg.place_axes_on_grid(fig, xspan=xspans[0], yspan=yspans[1]),  # D
        'KSregs': fg.place_axes_on_grid(fig, xspan=xspans[1], yspan=yspans[1]),  # E
        'c_labs': fg.place_axes_on_grid(fig, xspan=xspan_inset[0], yspan=yspans[1]),  # F
        'F_2': fg.place_axes_on_grid(fig, xspan=xspan_inset[1], yspan=yspans[1]),
        'm_labs': fg.place_axes_on_grid(fig, xspan=xspans_row3_1[0], yspan=yspans[2]),  # G
        'KSlabs': fg.place_axes_on_grid(fig, xspan=xspans_row3_1[1], yspan=yspans[2]),  # H
        'KS': fg.place_axes_on_grid(fig, xspan=xspans_row3_2[0], yspan=yspans[2]),  # I
        'KSmean': fg.place_axes_on_grid(fig, xspan=xspans_row3_2[1], yspan=yspans[2]),  # J
    }

    labels = [{'label_text': 'a', 'xpos': get_label_pos(width_m,xspans[0][0], pad=padx),
               'ypos': get_label_pos(height_m, yspans[0][0], pad=pady),
               'fontsize': 10, 'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'b', 'xpos': get_label_pos(width_m, xspans[1][0], pad=padx),
               'ypos': get_label_pos(height_m, yspans[0][0], pad=pady), 'fontsize': 10,
               'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'c', 'xpos': get_label_pos(width_m, xspans[2][0], pad=padx),
               'ypos': get_label_pos(height_m, yspans[0][0], pad=pady), 'fontsize': 10,
               'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'd', 'xpos': get_label_pos(width_m, xspans[0][0], pad=padx),
               'ypos': get_label_pos(height_m, yspans[1][0], pad=pady), 'fontsize': 10,
               'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'e', 'xpos': get_label_pos(width_m, xspans[1][0], pad=padx),
               'ypos': get_label_pos(height_m, yspans[1][0], pad=pady), 'fontsize': 10,
               'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'f', 'xpos': get_label_pos(width_m, xspans[2][0], pad=padx),
               'ypos': get_label_pos(height_m,yspans[1][0], pad=pady), 'fontsize': 10,
               'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'g', 'xpos': get_label_pos(width_m, xspans_row3_1[0][0], pad=padx),
               'ypos': get_label_pos(height_m, yspans[2][0], pad=pady), 'fontsize': 10,
               'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'h', 'xpos': get_label_pos(width_m, xspans_row3_1[1][0], pad=padx),
               'ypos': get_label_pos(height_m, yspans[2][0], pad=pady), 'fontsize': 10,
               'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'i', 'xpos': get_label_pos(width_m, xspans_row3_2[0][0], pad=0.2),
               'ypos': get_label_pos(height_m, yspans[2][0], pad=pady), 'fontsize': 10,
               'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'j', 'xpos': get_label_pos(width_m, xspans_row3_2[1][0], pad=-0.2),
               'ypos': get_label_pos(height_m, yspans[2][0], pad=pady), 'fontsize': 10,
               'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              ]

    fg.add_labels(fig, labels)

    # Supplementary figure
    width = 7
    height = 7
    figs = plt.figure(figsize=(width, height))

    xspans = get_row_coord(width, [1, 1, 1], hspace=0.8, pad=0.6)
    xspan_full = get_row_coord(width, [1], pad=0)
    yspans = get_row_coord(height, [8, 8, 8, 8, 8, 1], hspace=[0.8, 0.8, 0.8, 0.8, 0.6], pad=0.3)

    axss = {
        'Ha': fg.place_axes_on_grid(figs, xspan=xspans[0], yspan=yspans[0]),  # A
        'Hb': fg.place_axes_on_grid(figs, xspan=xspans[1], yspan=yspans[0]), # B
        'H': fg.place_axes_on_grid(figs, xspan=xspans[2], yspan=yspans[0]),  # C
        'Ia': fg.place_axes_on_grid(figs, xspan=xspans[0], yspan=yspans[1]), # D
        'Ib': fg.place_axes_on_grid(figs, xspan=xspans[1], yspan=yspans[1]),  # E
        'I': fg.place_axes_on_grid(figs, xspan=xspans[2], yspan=yspans[1]),  # F
        'Ja': fg.place_axes_on_grid(figs, xspan=xspans[0], yspan=yspans[2]), # G
        'Jb': fg.place_axes_on_grid(figs, xspan=xspans[1], yspan=yspans[2]),  # H
        'J': fg.place_axes_on_grid(figs, xspan=xspans[2], yspan=yspans[2]),  # I
        'Ka': fg.place_axes_on_grid(figs, xspan=xspans[0], yspan=yspans[3]), # J
        'Kb': fg.place_axes_on_grid(figs, xspan=xspans[1], yspan=yspans[3]),  # K
        'K': fg.place_axes_on_grid(figs, xspan=xspans[2], yspan=yspans[3]),  # L
        'Ga': fg.place_axes_on_grid(figs, xspan=xspans[0], yspan=yspans[4]), # M
        'Gb': fg.place_axes_on_grid(figs, xspan=xspans[1], yspan=yspans[4]),  # N
        'G': fg.place_axes_on_grid(figs, xspan=xspans[2], yspan=yspans[4]),  # O
        'Labs': fg.place_axes_on_grid(figs, xspan=xspan_full[0], yspan=yspans[5]),
    }

    labels = [{'label_text': 'a', 'xpos': get_label_pos(width,xspans[0][0], pad=0.6),
               'ypos': get_label_pos(height, yspans[0][0], pad=pady),
               'fontsize': 10, 'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'b', 'xpos': get_label_pos(width, xspans[1][0], pad=0.5),
               'ypos': get_label_pos(height, yspans[0][0], pad=pady), 'fontsize': 10,
               'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'c', 'xpos': get_label_pos(width, xspans[2][0], pad=0.5),
               'ypos': get_label_pos(height, yspans[0][0], pad=pady), 'fontsize': 10,
               'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'd', 'xpos': get_label_pos(width, xspans[0][0], pad=0.6),
               'ypos': get_label_pos(height, yspans[1][0], pad=pady), 'fontsize': 10,
               'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'e', 'xpos': get_label_pos(width, xspans[1][0], pad=0.5),
               'ypos': get_label_pos(height, yspans[1][0], pad=pady), 'fontsize': 10,
               'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'f', 'xpos': get_label_pos(width, xspans[2][0], pad=0.5),
               'ypos': get_label_pos(height,yspans[1][0], pad=pady), 'fontsize': 10,
               'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'g', 'xpos': get_label_pos(width, xspans[0][0], pad=0.6),
               'ypos': get_label_pos(height, yspans[2][0], pad=pady), 'fontsize': 10,
               'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'h', 'xpos': get_label_pos(width, xspans[1][0], pad=0.5),
               'ypos': get_label_pos(height, yspans[2][0], pad=pady), 'fontsize': 10,
               'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'i', 'xpos': get_label_pos(width, xspans[2][0], pad=0.5),
               'ypos': get_label_pos(height, yspans[2][0], pad=pady), 'fontsize': 10,
               'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'j', 'xpos': get_label_pos(width, xspans[0][0], pad=0.6),
               'ypos': get_label_pos(height, yspans[3][0], pad=pady), 'fontsize': 10,
               'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'k', 'xpos': get_label_pos(width, xspans[1][0], pad=0.5),
               'ypos': get_label_pos(height, yspans[3][0], pad=pady), 'fontsize': 10,
               'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'l', 'xpos': get_label_pos(width, xspans[2][0], pad=0.5),
               'ypos': get_label_pos(height, yspans[3][0], pad=pady), 'fontsize': 10,
               'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'm', 'xpos': get_label_pos(width, xspans[0][0], pad=0.6),
               'ypos': get_label_pos(height, yspans[4][0], pad=pady), 'fontsize': 10,
               'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'n', 'xpos': get_label_pos(width, xspans[1][0], pad=0.5),
               'ypos': get_label_pos(height, yspans[4][0], pad=pady), 'fontsize': 10,
               'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'o', 'xpos': get_label_pos(width, xspans[2][0], pad=0.5),
               'ypos': get_label_pos(height, yspans[4][0], pad=pady), 'fontsize': 10,
               'weight': 'bold', 'ha': 'right', 'va': 'bottom'}
              ]

    fg.add_labels(figs, labels)

    labs__ = Counter(labs)
    
    #order labs canonically
    labs_ = []

    for lab in canon:
        if lab in labs__:
            labs_.append(lab)

    '''
    ###
    perm tests, plot
    ###
    '''

    if perm_tests:
        # run all permutation tests
        perm_test(inclu=inclu, align=align,
                  nrand=nrand, # sig_lev =sig_lev,
                  fdr = fdr, ax=axs['KS'],samp=False, rerun=rerun) 
                  
        # average across subset sampling
        perm_test(inclu=inclu, align=align, nrand=nrand, #sig_lev =sig_lev,
                    fdr = fdr, ax=axs['KSmean'], samp=True,
                    nns = nns, rerun=rerun)
        axs['KSmean'].set_yticklabels([])
              
        # put panel label
        # for pan in ['KS']:
        #     axs[pan].text(-0.1, 1.3, panel_n[pan],
        #                     transform=axs[pan].transAxes,
        #                     fontsize=plfs, va='top',
        #                     ha='right', weight='bold')

    '''
    ###
    plot scatter, all cells, colored by reg
    ###
    '''
    
    

    # scatter 2d PCs
    cols_reg = [Dc[x] for x in regs]
    axs['B'].scatter(emb[:, 0], emb[:, 1], marker='o', c=cols_reg, s=2)
    
    # centers per region 
    regs__ = Counter(regs)
    regs_ = []
    for reg in canon_regs:
        if reg in regs__:
            regs_.append(reg)
        
    cents = {reg: np.mean(emb[regs == reg], axis=0)
             for reg in regs_}

    pos = np.linspace(0.2, 0.8, len(regs_))[::-1]
    for p, r in zip(pos, regs_):
        axs['C_2'].text(0.5, p, r, color=Dc[r], fontsize=7, transform=axs['C_2'].transAxes)

    axs['C_2'].set_axis_off()

    axs['B'].set_title('Regions')
    axs['B'].set_xlabel('Embedding dim 1')
    axs['B'].set_ylabel('Embedding dim 2')

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
    all_labs = np.copy(labs_)

    pos = np.linspace(0, 1, len(labs_))[::-1]
    institutes = [b[l] for l in labs_]
    institutes.sort()
    for p, l in zip(pos, institutes):
        axs['F_2'].text(0.5, p, l, color=lab_cols[l], fontsize=7, transform=axs['F_2'].transAxes)
    
    axs['F_2'].set_axis_off()

    axs['c_labs'].set_title('Labs')
    axs['c_labs'].set_xlabel('Embedding dim 1')
    axs['c_labs'].set_ylabel('Embedding dim 2')

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
        cellsinreg = len(y[regs == reg])

        ms = np.mean((y[regs == reg]) / T_BIN, axis=0)
        ste = np.std((y[regs == reg]) / T_BIN, axis=0) / np.sqrt(cellsinreg)

        # only plot one PSTH
        ms = ms[:len(xs)]
        ste = ste[:len(xs)]

        axs['D'].plot(xs, ms, color=Dc[reg])
        axs['D'].fill_between(xs, ms + ste, ms - ste, 
                              color=Dc[reg], alpha=0.2)

    for x in np.array([25]) * T_BIN:
        axs['D'].axvline(x=0, linestyle='--', c='k')

    axs['D'].set_xlabel(f'Time from {align} onset (s)')
    axs['D'].set_ylabel('Baselined firing rate (spikes/s)')


    '''
    ###
    # plot average PSTHs arcoss labs, all cells
    ###
    '''

    for lab in labs_:

        labsinlab = len(Counter(labs[labs == lab]))
        sessinlab = len(Counter(sess[labs == lab]))
        cellsinlab = len(y[labs == lab])

        ms = np.mean((y[labs == lab]) / T_BIN, axis=0)
        ste = np.std((y[labs == lab]) / T_BIN, axis=0) / np.sqrt(cellsinlab)

        # only plot one PSTH
        ms = ms[:len(xs)]
        ste = ste[:len(xs)]

        axs['m_labs'].plot(xs, ms, color=lab_cols[b[lab]])
        axs['m_labs'].fill_between(xs, ms + ste, ms - ste, 
                              color=lab_cols[b[lab]], alpha=0.2)

    for x in np.array([25]) * T_BIN:
        axs['m_labs'].axvline(x=0, linestyle='--', c='k')

    axs['m_labs'].set_xlabel(f'Time from {align} onset (s)')
    axs['m_labs'].set_ylabel('Baselined firing rate (spikes/s)')
    # axs['m_labs'].text(-0.1, 1.30, panel_n['m_labs'],
    #               transform=axs['m_labs'].transAxes, fontsize=plfs,
    #               va='top', ha='right', weight='bold')


    '''
    example cell reconstruction
    '''

    le = [Line2D([0], [0], color='g', lw=0.5, ls='--')]

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


        axs[ms[k]].plot(xs, y[idxs[k]][:len(xs)] / T_BIN,
                        c='k', label='PETH')
        axs[ms[k]].plot(xs, y_res[2][idxs[k]][:len(xs)] / T_BIN,
                        c='r', ls='--', label='2-PC-fit')

        axs[ms[k]].axvline(x=0, linestyle='--', c='k')

        stext = rf'$R^2$={np.round(r2_score(y[idxs[k]], y_res[2][idxs[k]]), 2)}'

        if k == 1:
            axs[ms[k]].set_xlabel(f'Time from {align} onset (s)')
            axs[ms[k]].set_ylabel('Baselined firing rate (spikes/s)')
            coords = axs[ms[k]].yaxis.get_label().get_position()
            print(coords)
            axs[ms[k]].yaxis.set_label_coords(-0.2, 1.2)
            axs[ms[k]].text(0.7, 0.1, stext, transform=axs[ms[k]].transAxes,
                            fontsize=7, va='bottom', ha='left')
        else:
            axs[ms[k]].text(0.7, 0.9, stext, transform=axs[ms[k]].transAxes,
                            fontsize=7, va='bottom', ha='left')
            axs[ms[k]].set_xticklabels([])

        # if k == 0:
        #     axs[ms[k]].text(-0.1, 1.6, panel_n[ms[k]],
        #                     transform=axs[ms[k]].transAxes,
        #                     fontsize=plfs, va='top',
        #                     ha='right', weight='bold')
                              
        k += 1

    for re_rank in r2s_:
        _, patches, _ = axs['F'].hist(x=r2s_[re_rank], 
                                      bins='auto', label=re_rank, 
                                      alpha=0.7, rwidth=0.85, 
                                      histtype=u'step', lw=2)

    leg = axs['F'].legend(ncol=1,frameon=False,
                          prop={'size': 6},
                          title="# PCs", fontsize=5).set_draggable(True)
                          
    axs['F'].set_xlabel(r'$r^2$')
    axs['F'].set_ylabel('Number of neurons')
    # axs['F'].text(-0.1, 1.30, panel_n['F'],
    #               transform=axs['F'].transAxes, fontsize=plfs,
    #               va='top', ha='right', weight='bold')


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
    # axs['KSlabs'].text(-0.1, 1.30, panel_n['KSlabs'],
    #                  transform=axs['KSlabs'].transAxes,
    #                  fontsize=plfs, va='top',
    #                  ha='right', weight='bold')

    # plot ks scores as bar plot inset with asterics for small p
    axsir = inset_axes(axs['KSlabs'], width="65%", height="35%",
                           borderpad=0, loc=4,
                           bbox_to_anchor=(0.45, 0.2, 1, 1),
                           bbox_transform=axs['KSlabs'].transAxes)
               
    bars = axsir.bar(range(len(ksr)), [ksr[lab][0] for lab in ksr], 
                  color = [lab_cols[b[lab]] for lab in ksr])
                  
    axsir.set_xlabel('Labs')
    axsir.set_xticks([])
    axsir.set_xticklabels([])
    axsir.set_ylabel('KS')
    # axsir.spines['top'].set_visible(False)
    # axsir.spines['right'].set_visible(False)
                 
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
    # axs['KSregs'].text(-0.1, 1.30, panel_n['KSregs'],
    #                  transform=axs['KSregs'].transAxes,
    #                  fontsize=plfs, va='top',
    #                  ha='right', weight='bold')

    # plot ks scores as bar plot inset with asterics for small p 
    axsir = inset_axes(axs['KSregs'], width="30%", height="35%", 
                           loc=4, borderpad=0,
                           bbox_to_anchor=(0.1,0.2,1,1),
                           bbox_transform=axs['KSregs'].transAxes)
               
    bars = axsir.bar(range(len(ksr)), [ksr[reg][0] for reg in ksr], 
                  color = [Dc[reg] for reg in ksr])
                  
    axsir.set_xlabel('Regions')
    axsir.set_xticks([])
    axsir.set_xticklabels([])
    axsir.set_ylabel('KS')
    # axsir.spines['top'].set_visible(False)
    # axsir.spines['right'].set_visible(False)
                 
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
    axs['KSregs'].set_xlim(-2, 6)


    '''
    analysis per region
    '''
    
    ms = ['H', 'I', 'J', 'K','G']  # 2PCs scatter 'G',
    ms2 = ['Ha', 'Ia', 'Ja', 'Ka','Ga']  #'Ga',  average PETH
    ms3 = ['Hb', 'Ib', 'Jb', 'Kb','Gb']  #'Gb',  CDFs

    k = 0
    p_ = {}  # Guido's permutation test score
    p_ks = {}  # KS test on 1st PC
    D = canon_regs 
    
    axsi = []  # inset axes
    for reg in D:

        if reg is None:
            continue

        axs3 = axss
        # panel_n3 = panel_ns

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
        

        # axs3[ms3[k]].set_title(reg, loc='left')
        axs3[ms3[k]].set_xlabel('x')
        axs3[ms3[k]].set_ylabel('P(PC1 < x)')

        if k == 1:
            axs3[ms3[k]].legend(frameon=False, 
                                loc='upper left').set_draggable(True)

        # plot ks scores as bar plot inset with asterics for small p
        axsi.append(axs3[ms3[k]].inset_axes((0.7, 0.3, 0.3, 0.35),
                               transform=axs3[ms3[k]].transAxes))
                   
        bars = axsi[k].bar(range(len(ksr)), [ksr[lab][0] for lab in ksr], 
                      color = [lab_cols[b[lab]] for lab in ksr])
                      
        axsi[k].set_xlabel('Labs')
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

        axs3[ms[k]].set_xlabel('Embedding dim 1')
        axs3[ms[k]].set_ylabel('Embedding dim 2')

        # plot average PSTHs across labs
        for lab in labs_:

            mes = np.mean((y2[labs2 == lab]) / T_BIN, axis=0)
            # normalise by the number of sessions, not the number of cells

            sessinlab = len(Counter(sess2[labs2 == lab]))
            cellsinlab = len(y[labs == lab])

            ste = np.std((y2[labs2 == lab]) / T_BIN, axis=0) / np.sqrt(cellsinlab)
            mes = mes[:len(xs)]
            ste = ste[:len(xs)]

            axs3[ms2[k]].plot(xs, mes, color=lab_cols[b[lab]])
            axs3[ms2[k]].fill_between(xs, mes + ste, mes - ste,
                                      color=lab_cols[b[lab]], alpha=0.2)

        axs3[ms2[k]].set_title(reg, loc='right')
        axs3[ms2[k]].set_xlabel(f'Time from {align} onset (s)')
        axs3[ms2[k]].set_ylabel('Baselined firing rate \n (spikes/s)')
        
        axs3[ms2[k]].axvline(x=0, linestyle='--', c='k')

        # axs3[ms2[k]].text(-0.1, 1.30, panel_n3[ms2[k]],
        #                   transform=axs3[ms2[k]].transAxes,
        #                   fontsize=plfs, va='top', ha='right', weight='bold')

        le = [Line2D([0], [0], color='g', lw=0.5, ls='--')]

        axs3[ms[k]].sharex(axs['B'])
        axs3[ms[k]].sharey(axs['B'])
        axs3[ms2[k]].sharex(axs['D'])
        axs3[ms2[k]].sharey(axs['D'])

        k += 1

    axss['Labs'].set_axis_off()

    institutes = [b[l] for l in all_labs]
    institutes.sort()
    for i, l in enumerate(institutes):
        if i == 0:
            text = axss['Labs'].text(0.2, 0.5, l, color=lab_cols[l], fontsize=8, transform=axss['Labs'].transAxes)
        else:
            text = axss['Labs'].annotate(
                '  ' + l, xycoords=text, xy=(1, 0), verticalalignment="bottom",
                color=lab_cols[l], fontsize=8)  # custom properties


    axs['B'].set_xlim([-2, 1.5])
    axs['B'].set_ylim([-1.1, 1.1])
    axs['D'].set_xlim([-0.2, 0.8])
    axs['D'].set_ylim([-3, 10])

    # shift x position of KSshift heatmap to correct for long text
    # B = axs['KS'].get_position()
    # axs['KS'].set_position([B.x0 -0.02, B.y0, B.width, B.height])

    fig_path = save_figure_path(figure='fig_PCA')
    print(f'Saving figures to {fig_path}')
    adjust = 0.3
    fig.subplots_adjust(top=1-adjust/height_m, bottom=(adjust + 0.2)/height_m, left=(adjust)/width_m, right=1-(adjust + 0.2)/width_m)
    fig.savefig(fig_path.joinpath('figure_PCA.pdf'))

    adjust = 0.3
    figs.subplots_adjust(top=1-adjust/height, bottom=(adjust - 0.1)/height, left=(adjust)/width, right=1-(adjust)/width)
    figs.savefig(fig_path.joinpath('figure_PCA_supp1.pdf'))


def linear_mixed_effects(align='move', comp='pc1'):

    emb00, regs00, labss00, eids00 = all_panels(get_dat=True, align=align)

    df00 = pd.DataFrame({
                        'pc1': emb00[:,0],
                        'pc2': emb00[:,1],
                        'regs': regs00,
                        'labs': labss00,
                        'eids': eids00})

    model = smf.mixedlm(f"{comp} ~ regs + labs", data=df00, 
                        groups="eids", re_formula="1")

    return model.fit(method='cg')


def plot_SI_lme(fdr=True):
    '''
    For alignments ['move', 'stim'] and components ['pc1', 'pc2'],
    plot coefficients of linear mixed effects model in a 1x4 grid.
    Each panel shows coefficients with error bars and 95% confidence intervals.
    fdr: fasle discovery rate correction, at sigl=0.05
    '''
    
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(7.14, 3), sharey=True)
    plt.ion()

    alignments = ['move', 'stim']
    components = ['pc1', 'pc2']
    pans = [(a, c) for a in alignments for c in components]

    if fdr:
        D = {}
        for (align, comp) in pans:

            result = linear_mixed_effects(align=align, comp=comp)
            plot_df = pd.DataFrame(result.summary().tables[1])
            # Convert necessary columns to numeric
            plot_df[['Coef.', 'Std.Err.', 'z', 'P>|z|', '[0.025', '0.975]']] = plot_df[
                ['Coef.', 'Std.Err.', 'z', 'P>|z|', '[0.025', '0.975]']
            ].apply(pd.to_numeric, errors='coerce')
            D[f'{align}_{comp}'] = plot_df

        ps = np.concatenate([D[key]['P>|z|'].values for key in D])
        ps[np.isnan(ps)] = 1
        n_p = len(D[list(D.keys())[0]]['P>|z|'])
        _, ps_c, _, _ = multipletests(ps, sig_lev, method='fdr_bh')

        for k in range(len(D)):
            D[list(D.keys())[k]]['P>|z|'] = ps_c[k*n_p:(k+1)*n_p] 
            k+=1
        print(f'fdr corrected p-values at {sig_lev}')

    for idx, (align, comp) in enumerate(pans):
        ax = axs[idx]

        if fdr:
            plot_df = D[f'{align}_{comp}'] 
        else:
            result = linear_mixed_effects(align=align, comp=comp)
            plot_df = pd.DataFrame(result.summary().tables[1])

            # Convert necessary columns to numeric
            plot_df[['Coef.', 'Std.Err.', 'z', 'P>|z|', '[0.025', '0.975]']] = plot_df[
                ['Coef.', 'Std.Err.', 'z', 'P>|z|', '[0.025', '0.975]']
            ].apply(pd.to_numeric, errors='coerce')


        # Define colors and error bars
        for row_idx, row in plot_df.iterrows():
            coef = row['Coef.']
            lower_bound = row['[0.025']
            upper_bound = row['0.975]']
            color = 'red' if row['P>|z|'] < sig_lev else 'grey'
            
            ax.errorbar(
                coef, row_idx, 
                xerr=[[coef - lower_bound], [upper_bound - coef]], 
                fmt='o', color=color , ecolor=color, elinewidth=2, capsize=0
            )

        # Set y-axis labels with colors for lab or region
        y_labels = plot_df.index
        y_colors = [
            'black' if label in ['Intercept', 'eids Var']
            else lab_cols[b[label.split('.')[1][:-1]]]
            if 'labs' in label else Dc[label.split('.')[1][:-1]]
            for label in y_labels
        ]

        ax.set_yticks(range(len(y_labels)))
        for tick, color in zip(ax.get_yticklabels(), y_colors):
            tick.set_color(color)

        # Title and labels
        ax.set_title(f"{align.capitalize()}, {comp.upper()}")
        if idx == 0:
            ax.set_yticks(range(len(plot_df)))
            ax.set_yticklabels(plot_df.index)

        ax.axhline(y=4.5, color='black', linestyle='--')
        ax.axvline(x=0, color='gray', linestyle='--')
        
        ax.set_xlabel("Coefficient Estimate")

    plt.tight_layout()
    plt.show()

import numpy as np
from collections import Counter

from scipy.stats import ks_2samp
import random
from copy import deepcopy
import pandas as pd

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

from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from datetime import datetime

from reproducible_ephys_functions import filter_recordings, save_figure_path, labs, figure_style
from fig_PCA.fig_PCA_load_data import load_dataframe, load_data

import warnings
warnings.filterwarnings("ignore")

T_BIN = 0.02  # time bin size in seconds
_, b, lab_cols = labs()

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


def all_panels(rm_unre=True, align='move', split='rt', 
               xyz_res=False, re_rank=2, fdr=True, permute_include=True,
               min_lab = False, show_total = False, nrand = 1000,
               sig_lev = 0.01):
                             
    '''
    min_lab: restrict to two areas so that all labs have same #cells
    show_total: in black plot mean and ellipse of all cells
    '''
               
    # load metainfo df, row per cell
    concat_df = load_dataframe()
    # load PSTHs, one per cell
    data = load_data(event=align, split=split, smoothing='none')
    all_frs = data['all_frs']

    ts = 'fast|slow RT PETH'

    # include has minimum number of clusters as being 3
    concat_df = filter_recordings(concat_df)

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
    xyz = np.array([concat_df[g].values for g in ['x', 'y', 'z']]).T

    # merge hofer and mirsicflogel labs
    labs[labs == 'mrsicflogellab'] = 'hoferlab'

    if min_lab:
      
        # sample cells to have same number from each lab                
        labC = Counter(labs)
        lr = {lab : dict(Counter(regs[labs == lab])) for lab in labC}
        creg = set.intersection(*[set(np.unique(regs[labs == lab])) 
                                 for lab in labC])                                 
        
        idc = np.arange(len(regs))                         
        idc_ = []
        for reg in creg:
            mreg = min([lr[lab][reg] for lab in lr])
            # sample mreg cells from each lab for region reg
            for lab in lr:
                idc_.append(random.sample(
                            list(idc[np.bitwise_and(
                            labs == lab, regs == reg)]), mreg))

        idc_ = list(np.concatenate(idc_))
        
        # restric data to regs/labs have same #cells
        y = y[idc_]
        regs = regs[idc_]
        labs = labs[idc_]
        xyz = xyz[idc_]

    # PCA embedding
    pca = PCA(n_components=2)
    pca.fit(y)
    emb = pca.transform(y)


    if xyz_res:
        clf = Ridge(alpha=0)
        clf.fit(xyz, emb)
        emb_p = clf.predict(xyz)
        res = emb_p - emb
        emb = res
        print('xyz influence subtracted')

    # get reproduction using serveral first PCs
    u, s, vh = np.linalg.svd(y)
    y_res = {}

    for re_rank in [1, 2, 3]:
        S_star = np.zeros(y.shape)
        for i in range(re_rank):
            S_star[i, i] = s[i]
        y_res[re_rank] = np.array(np.matrix(u) * np.matrix(S_star) * np.matrix(vh))

    xs = np.arange(len(y[0])) * T_BIN
    xs = xs[:len(xs) // 2]  # for just plotting first PSTH

    # get a sessions idx per cell (for stats model)
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

    mosaic = [[inner, 'F'],
              ['B', 'D'],
              ['G', 'Ga'],
              ['c_reg', 'Gb']]

    mosaic_supp = [['Ha', 'Hb', 'H'],
                   ['Ia', 'Ib', 'I'],
                   ['Ja', 'Jb', 'J'],
                   ['Ka', 'Kb', 'K'],
                   ['Ga', 'Gb', 'G']]
                   
    mf = np.array(mosaic, dtype=object).flatten()
    mf[0] = 'Ea'
    panel_n = dict(zip(mf, string.ascii_lowercase))

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
    # Euclidean distance of points for permutation test
    #return labs_
    def distE(x, y):
        return np.sqrt(np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y))

    le_labs = [Patch(facecolor=lab_cols[b[lab]], 
               edgecolor=lab_cols[b[lab]], label=b[lab]) for lab in labs_]
               
    if show_total:
        le_labs.append(Patch(facecolor='k',
                       edgecolor='k', label='all cells'))

    axs['G'].legend(handles=le_labs, loc='lower left', 
                    bbox_to_anchor=(0.1, 1), ncol=3, frameon=False, 
                    prop={'size': 8}).set_draggable(True)

    regs_ = Counter(regs)
    cents = {}
    for reg in regs_:
        cents[reg] = np.mean(emb[regs == reg], axis=0)


    # shuffle test 
    if nrand < 1000: # random region allocations
        print('put nrand back to 1000')
        
    centsr = []
    for shuf in range(nrand):
        regsr = regs.copy()
        random.shuffle(regsr)
        cenr = {}
        for reg in regs_:
            cenr[reg] = np.mean(emb[regsr == reg], axis=0)
        centsr.append(cenr)

    # Guido's permutation test
    cs = np.mean([cents[l] for l in cents], axis=0)
    dist = sum([distE(cents[l], cs) for l in cents])

    null_d = [sum([distE(cenr[reg],
              np.mean([cenr[l] for l in cenr], axis=0)) for reg in cenr])
              for cenr in centsr]

    p = np.mean(np.array(null_d + [dist]) >= dist) 
    print('region permutation test:', p)

    '''
    ###
    plot scatter, all labs, colored by reg
    ###
    '''
    Dc = figure_style(return_colors=True)

    # scatter 2d PCs
    cols_reg = [Dc[x] for x in regs]
    axs['B'].scatter(emb[:, 0], emb[:, 1], marker='o', c=cols_reg, s=2)
        
    
    for reg in cents:
        # plot means
        axs['B'].scatter(cents[reg][0], cents[reg][1], 
                         s=500, marker='x', color=Dc[reg])
                         
        # plot confidence ellipses
        x = emb[regs == reg]
        confidence_ellipse(x[:, 0], x[:, 1], axs['B'], 
                           n_std=1.0, edgecolor=Dc[reg])        

    le = [Patch(facecolor=Dc[reg], edgecolor=Dc[reg], 
                label=reg) for reg in regs_]

    if show_total:
        # plot mean of all points and ellipse
        axs['B'].scatter(cs[0], cs[1], 
                         s=500, marker='x', color='k')
           
        confidence_ellipse(emb[:, 0], emb[:, 1], axs['B'], 
                           n_std=1.0, edgecolor='k') 
                           
        le.append(Patch(facecolor='k', edgecolor='k',
                        label='all cells'))                       


    # additional test, instead of distance of means, KS of distrbs
    # KS distance between 1 PCs of cells of one region to that of all other regions
        
    g_all = emb[:,0]
    g_regs = {}
    for reg in regs_:
        g_regs[reg] = emb[regs == reg][:,0]
        
    dist = sum([ks_2samp(g_all, g_regs[reg])[0] for reg in g_regs]) 
    
    # get null_d by shuffling region labels
    null_d = []
    for shuf in range(nrand):
        regsr = regs.copy()
        random.shuffle(regsr)
        g_regs = {}
        for reg in regs_:
            g_regs[reg] = emb[regsr == reg][:,0]    
        null_d.append(sum([ks_2samp(g_all, g_regs[reg])[0] 
                           for reg in g_regs]))    

    p = np.mean(np.array(null_d + [dist]) >= dist) 
    print('region permutation test (KS using 1st PC):', p)


    axs['B'].legend(handles=le, bbox_to_anchor=(0.3, 1), 
                    loc='lower left', ncol=3, frameon=False,
                    prop={'size': 7}).set_draggable(True)

    #axs['B'].set_title('all cells', loc='left')
    axs['B'].set_xlabel('embedding dim 1')
    axs['B'].set_ylabel('embedding dim 2')
    axs['B'].text(-0.1, 1.30, panel_n['B'], 
                  transform=axs['B'].transAxes, fontsize=16, 
                  va='top', ha='right', weight='bold')

    axs['B'].sharex(axss['H'])
    axs['B'].sharey(axss['H'])
    axs['D'].sharex(axs['Ga'])
    axs['D'].sharey(axs['Ga'])


    '''
    ###
    plot 1st PC and KS for all labs, split by region
    ###
    '''
    # 1PCs and comparing distributions via KS - by regions 
    
    # plot CDFs of first PCs distributions and KS metrics

    x0, x1 = ecdf(g_all)
    axs['c_reg'].plot(x0, x1,
                   label=f'all cells', lw=2,
                   color = 'k', drawstyle='steps-post')
                   
           
    ksr = {}                          
    for reg in regs_:
        x0, x1 = ecdf(g_regs[reg])
        ks,p = ks_2samp(g_all, g_regs[reg])
        ksr[reg] = [ks,p]
        axs['c_reg'].plot(x0, x1, color=Dc[reg], 
                   drawstyle='steps-post', lw=1)
        #print(reg, ks, p)          
                   
    
    # per reg multiple comparison corr                
    pvals_ = [ksr[reg][1] for reg in ksr]
    if fdr:
        _, pvals_c_, _, _ = multipletests(pvals_, sig_lev, method='fdr_bh')
    else:
        pvals_c_ = pvals_
    
    
    print(f'KS results: is region different from all cells? \n'
          '[reg, k-statistic, p-value]')        
    kk = 0
    if fdr:
        print('corrected')
    else:
        print('uncorrected')      
    for reg in ksr:
        ksr[reg][1] = pvals_c_[kk]
        kk += 1
        
#        if ksr[reg][1] < sig_lev:
  
        print(reg, ksr[reg][0], ksr[reg][1])
                   
    axs['c_reg'].set_title('all cells', loc='left')
    axs['c_reg'].set_xlabel('PC1')
    axs['c_reg'].set_ylabel('P(PC1 < x)')
    axs['c_reg'].text(-0.1, 1.30, panel_n['c_reg'],
                     transform=axs['c_reg'].transAxes,
                     fontsize=16, va='top',
                     ha='right', weight='bold')            

    axs['c_reg'].legend(frameon=False, 
                        loc='upper left').set_draggable(True)

    # plot ks scores as bar plot inset with asterics for small p 
    axsi0 = inset_axes(axs['c_reg'], width="30%", height="35%", 
                           loc=4, borderpad=1,
                           bbox_to_anchor=(-0.02,0.1,1,1), 
                           bbox_transform=axs['c_reg'].transAxes)
               
    bars = axsi0.bar(range(len(ksr)), [ksr[reg][0] for reg in ksr], 
                  color = [Dc[reg] for reg in ksr])
                  
    axsi0.set_xlabel('labs')
    axsi0.set_xticks([])
    axsi0.set_xticklabels([])
    axsi0.set_ylabel('KS')
    axsi0.spines['top'].set_visible(False)
    axsi0.spines['right'].set_visible(False)
                 
    # put * on bars that are significant
    ii = 0
    for reg in ksr:
        ba = bars.patches[ii]
        if ksr[reg][1] < sig_lev:             
            axsi0.annotate('*',
                           (ba.get_x() + ba.get_width() / 2,
                            ba.get_height()), ha='center', 
                            va='center', size=16, 
                            xytext=(0, 1),
                            textcoords='offset points')
        ii += 1

    '''
    ###
    # plot average PSTHs
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

    axs['D'].set_xlabel('time from movement onset (s)')
    axs['D'].set_ylabel('Firing rate \n (spikes/s)')
    axs['D'].text(-0.1, 1.30, panel_n['D'], 
                  transform=axs['D'].transAxes, fontsize=16, 
                  va='top', ha='right', weight='bold')

    le = [Line2D([0], [0], color='g', lw=0.5, ls='--', 
                 label='movement onset')]

#    axs['D'].legend(handles=le, bbox_to_anchor=(0.3, 1), frameon=False, 
#                    loc='lower left', ncol=1, prop={'size': 7}).set_draggable(True)

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
        '''
        example cells PETH and 2-PC fit
        '''

        axs[ms[k]].plot(xs - 0.5, y[idxs[k]][:len(xs)] / T_BIN,
                        c='k', label='PETH')
        axs[ms[k]].plot(xs - 0.5, y_res[2][idxs[k]][:len(xs)] / T_BIN,
                        c='r', ls='--', label='2-PC-fit')

        for x in np.array([25]) * T_BIN:
            axs[ms[k]].axvline(x=0, linewidth=0.5, linestyle='--',
                               c='g', label='movement onset')

        axs[ms[k]].set_ylabel('Firing rate \n (spikes/s)')
        stext = rf'$r^2$={np.round(r2_score(y[idxs[k]], y_res[2][idxs[k]]), 2)}'
        axs[ms[k]].text(0.2, 1, stext, transform=axs[ms[k]].transAxes,
                        fontsize=7, va='top', ha='right')

        if k == 1:
            axs[ms[k]].set_xlabel('time from movement onset (s)')

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

    leg = axs['F'].legend(ncol=len(r2s_[re_rank]) / 
                          4,frameon=False,
                          prop={'size': 7}, 
                          title="number of PCs").set_draggable(True)
                          
    axs['F'].set_xlabel(r'$r^2$')
    axs['F'].set_ylabel('number of neurons')
    axs['F'].text(-0.1, 1.30, panel_n['F'], 
                  transform=axs['F'].transAxes, fontsize=16, 
                  va='top', ha='right', weight='bold')

    # per region dim red
    ms = ['G', 'H', 'I', 'J', 'K','G']  # 2PCs scatter
    ms2 = ['Ga', 'Ha', 'Ia', 'Ja', 'Ka','Ga']  # average PETH
    ms3 = ['Gb', 'Hb', 'Ib', 'Jb', 'Kb','Gb']  # CDFs

    k = 0
    p_ = {}  # Guido's permutation test score
    p_ks = {}  # KS test on 1st PC
    if min_lab:
        D = ['CA1', 'LP']
    else:
        D = ['CA1', 'PPC', 'CA1', 'DG', 'LP', 'PO']
    nns = {}
    
    axsi = []  # inset axes
    for reg in D:
        if reg is None:
            continue

        if k == 0:
            axs3 = axs
            panel_n3 = panel_n
        else:
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

        nnl = {}
        cents = {}
        for lab in labs_:
            cents[lab] = np.mean(emb2[labs2 == lab], axis=0)
            nnl[lab] = len(emb2[labs2 == lab])
        nns[reg] = nnl    
            
        # plot means
        for lab in cents:
            axs3[ms[k]].scatter(cents[lab][0], cents[lab][1], 
                                s=500, marker='x', color=lab_cols[b[lab]])

        # plot confidence ellipses
        for lab in labs_:
            x = emb2[labs2 == lab]
            confidence_ellipse(x[:, 0], x[:, 1], axs3[ms[k]], 
                               n_std=1.0, edgecolor=lab_cols[b[lab]])

        if show_total:
            cs = np.mean(emb2, axis=0)
            # plot mean of all points and ellipse
            axs3[ms[k]].scatter(cs[0], cs[1], 
                             s=500, marker='x', color='k')
               
            confidence_ellipse(emb2[:, 0], emb2[:, 1], axs3[ms[k]], 
                               n_std=1.0, edgecolor='k') 
                               

        # shuffle test
        centsr = []
        for shuf in range(nrand):
            labsr = labs2.copy()
            random.shuffle(labsr)
            cenr = {}
            for lab in labs_:
                cenr[lab] = np.mean(emb2[labsr == lab], axis=0)
            centsr.append(cenr)

        cs = np.mean([cents[lab] for lab in cents], axis=0)
        dist = sum([distE(cents[lab], cs) for lab in cents])

        null_d = [sum([distE(cenr[lab], 
                       np.mean([cenr[la] for la in cenr], axis=0)) 
                       for lab in cenr]) for cenr in centsr]
                       
        p = np.mean(np.array(null_d + [dist]) >= dist) 
        p_[reg] = p


        # additional test using 1PCs and comparing distributions via KS 
        g_all = emb2[:,0]
        g_labs = {}
        for lab in labs_:
            g_labs[lab] = emb2[labs2 == lab][:,0]

        
        # plot CDFs of first PCs distributions and KS metrics
        #if k != 0:
        x0, x1 = ecdf(g_all)
        axs3[ms3[k]].plot(x0, x1,
                       label=f'all cells', lw=2,
                       color = 'k', drawstyle='steps-post')
        ksr = {}                          
        for lab in labs_:
            x0, x1 = ecdf(g_labs[lab])
            ks,p = ks_2samp(g_all, g_labs[lab])
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
        if k == 1: print(f'KS results if p < {sig_lev}:')
        
        for lab in ksr:
            ksr[lab][1] = pvals_c_[kk]
            kk += 1
            
            if ksr[lab][1] < sig_lev:
                if fdr:
                    print('corrected')
                else:
                    print('uncorrected')    
                print(reg, lab, ksr[lab][0], ksr[lab][1])
        
        
        
        # extra KS test, custum p value:
        dist = sum([ks_2samp(g_all, g_labs[lab])[0] for lab in g_labs]) 
  

        # get null_d by shuffling lab labels
        null_d = []
        for shuf in range(nrand):
            labsr = labs2.copy()
            random.shuffle(labsr)
            g_labsr = {}
            for lab in labs_:
                g_labsr[lab] = emb2[labsr == lab][:,0]    
            null_d.append(sum([ks_2samp(g_all, g_labsr[lab])[0] 
                               for lab in g_labsr]))    
        
        p = np.mean(np.array(null_d + [dist]) >= dist) 
        print(f'reg {reg}, lab sum permutation test (KS using 1st PC):', p) 

                       
        axs3[ms3[k]].set_title(reg, loc='left')
        axs3[ms3[k]].set_xlabel('PC1')
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
                axsi[k-1].annotate('*',
                                   (ba.get_x() + ba.get_width() / 2,
                                    ba.get_height()), ha='center', 
                                    va='center', size=16, 
                                    xytext=(0, 1),
                                    textcoords='offset points')
            ii += 1
                
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
        axs3[ms2[k]].set_xlabel('time from movement onset (s)')
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

    pvals_ = [p_[reg] for reg in p_]
    
    if fdr:
        _, pvals_c_, _, _ = multipletests(pvals_, sig_lev, method='fdr_bh')
        print('means permutation test labs per region, corrected:')
    else:
        pvals_c_ = pvals_
        print('means permutation test labs per region, uncorrected:')
    
    
    dp = dict(zip(list(p_.keys()), [p for p in pvals_c_]))
    print(dp)
#    pvals_ks = [p_ks[reg] for reg in p_]
#    _, pvals_c_ks, _, _ = multipletests(pvals_ks, 0.05, method='fdr_bh')

#    print('KS permutation test labs per region, corrected:')
#    print(dict(zip(list(p_ks.keys()), [np.round(p, 3) for p in pvals_c_ks])))

    # print numbers of cells per region per lab
    #print(nns)

    axs['B'].set_xlim([-2, 1.5])
    axs['B'].set_ylim([-1.1, 1.1])
    axs['D'].set_xlim([-0.2, 0.8])
    axs['D'].set_ylim([-7, 12])

    fig.tight_layout()
    figs.tight_layout()

    if not min_lab:
        fig_path = save_figure_path(figure='fig_PCA')
        print(f'Saving figures to {fig_path}')
        fig.savefig(fig_path.joinpath('figure_PCA.pdf'))
        figs.savefig(fig_path.joinpath('figure_PCA_supp1.pdf'))
        
    #return dp
    
    
    
    
def all_dim_reds(rm_unre=True, align='move', split='rt', 
               xyz_res=False, re_rank=2, fdr=True, permute_include=True,
               show_total = False, nrand = 1000, sig_lev = 0.01,
               single = True):
                             
    '''
    min_lab: restrict to two areas so that all labs have same #cells
    show_total: in black plot mean and ellipse of all cells
    '''
               
    # load metainfo df, row per cell
    concat_df = load_dataframe()
    # load PSTHs, one per cell
    data = load_data(event=align, split=split, smoothing='none')
    all_frs = data['all_frs']

    ts = 'fast|slow RT PETH'

    # include has minimum number of clusters as being 3
    concat_df = filter_recordings(concat_df)

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
    xyz = np.array([concat_df[g].values for g in ['x', 'y', 'z']]).T

    # merge hofer and mirsicflogel labs
    labs[labs == 'mrsicflogellab'] = 'hoferlab'


    # PCA embedding
    pca = PCA(n_components=2)
    pca.fit(y)
    emb = pca.transform(y)    
    
    emb_u = umap.UMAP(n_components=2).fit_transform(y)
    emb_t = TSNE(n_components=2).fit_transform(y)

    # lab colors
    labs_ = Counter(labs)
    cols_lab = [lab_cols[b[x]] for x in labs]
    le_labs = [Patch(facecolor=lab_cols[b[lab]],
               edgecolor=lab_cols[b[lab]], label=b[lab]) for lab in labs_]
    
    # region colors
    regs_ = Counter(regs)
    Dc = figure_style(return_colors=True)
    cols_reg = [Dc[x] for x in regs]
    le = [Patch(facecolor=Dc[reg], edgecolor=Dc[reg], 
                label=reg) for reg in regs_]
    
    
    fig, axs = plt.subplots(nrows=2,ncols=3, figsize=(15,10))
    dr = {'tSNE': emb_t, 'umap': emb_u, 'PCA': emb}
    
    c = 0
    for m in dr:
        axs[0,c].scatter(dr[m][:, 0], dr[m][:, 1], marker='o', c=cols_reg, s=2)
        axs[1,c].scatter(dr[m][:, 0], dr[m][:, 1], marker='o', c=cols_lab, s=2)
        axs[1,c].sharex(axs[0,c])
        axs[1,c].sharey(axs[0,c])
        if c == 0:
            axs[0,c].legend(handles=le, loc='lower left',
                          bbox_to_anchor=(0.1, 1), ncol=3, frameon=False,
                          prop={'size': 8}).set_draggable(True)
            axs[1,c].legend(handles=le_labs, loc='lower left',
                          bbox_to_anchor=(0.1, 1), ncol=3, frameon=False,
                          prop={'size': 8}).set_draggable(True)
        
        
        axs[0,c].set_title(m)
        c+=1
    
    fig.tight_layout()
    fig_path = save_figure_path(figure='fig_PCA_supp')
    fig.savefig(fig_path.joinpath(f'all_regs_labs.png'))
    plt.close()    
      
    
    if single:
        plt.ioff()
        for lab in labs_:
            fig, axs = plt.subplots(nrows=1,ncols=3, num=lab, figsize=(15,10))
            
            dr = {'tSNE': emb_t, 'umap': emb_u, 'PCA': emb}
            
            c = 0
            for m in dr:
                axs[c].scatter(dr[m][:, 0], dr[m][:, 1], marker='o', 
                                 color='gray', s=4)
                axs[c].scatter(dr[m][labs==lab, 0], dr[m][labs==lab, 1], marker='o', 
                                 color='r', s=4)

                axs[c].set_title(m)
                c+=1
                
            fig.suptitle(lab)
            fig.tight_layout()    
            fig_path = save_figure_path(figure='fig_PCA_supp')
            fig.savefig(fig_path.joinpath(f'lab_{lab}.png'))
            plt.close()
            
            
        for reg in regs_:
            fig, axs = plt.subplots(nrows=1,ncols=3, num=reg, figsize=(15,10))
            
            dr = {'tSNE': emb_t, 'umap': emb_u, 'PCA': emb}
            
            c = 0
            for m in dr:
                axs[c].scatter(dr[m][:, 0], dr[m][:, 1], marker='o', 
                                 color='gray', s=4)
                axs[c].scatter(dr[m][regs==reg, 0], dr[m][regs==reg, 1], 
                               marker='o', color='r', s=4)

                axs[c].set_title(m)
                c+=1
            
            fig.suptitle(reg)
            fig.tight_layout()    
            fig_path = save_figure_path(figure='fig_PCA_supp')
            fig.savefig(fig_path.joinpath(f'reg_{reg}.png'))    
            plt.close()


def get_data(dim_red=False, rm_unre=True, align='move', split='rt', 
             re_rank=2, permute_include=True):

    # load metainfo df, row per cell
    concat_df = load_dataframe()
    # load PSTHs, one per cell
    data = load_data(event=align, split=split, smoothing='none')
    all_frs = data['all_frs']

    ts = 'fast|slow RT PETH'

    # include has minimum number of clusters as being 3
    concat_df = filter_recordings(concat_df)

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
    xyz = np.array([concat_df[g].values for g in ['x', 'y', 'z']]).T

    # merge hofer and mirsicflogel labs
    labs[labs == 'mrsicflogellab'] = 'hoferlab'


    # PCA embedding
    pca = PCA(n_components=2)
    pca.fit(y)
    emb = pca.transform(y)    

    return y, regs, labs, emb


def decode(x,y,decoder='LDA', CC = 1.0, shuf = False):
       
    #x,y = 
    

    print('input dimension:', np.shape(x)) 
    print('chance at ', 1/len(Counter(y)))

    startTime = datetime.now()

     
    if shuf:
        print('labels are SHUFFLED')
        np.random.shuffle(y)
        
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)

    train_r2 = []
    test_r2 = []
    acs = [] 
    
    y_p = []
    y_t = []    
    ws = []
    
    # cross validation    

    folds = 5
    kf = KFold(n_splits=folds, shuffle=True)
                                       
    k=1        
    for train_index, test_index in kf.split(x):
 
        sc = StandardScaler()
        train_X = sc.fit_transform(x[train_index])
        test_X = sc.fit_transform(x[test_index])

        train_y = y[train_index]
        test_y = y[test_index]    


        if k==1: 

            print('train/test samples:', len(train_y), len(test_y))
            print('')   

        if decoder == 'LR':
            clf = LogisticRegression(random_state=0,
                                     max_iter=200, n_jobs = -1)
            clf.fit(train_X, train_y)
            
            y_pred_test = clf.predict(test_X) 
            y_pred_train = clf.predict(train_X)  
   
        elif decoder == 'LDA':    
        
            clf = LinearDiscriminantAnalysis()
            clf.fit(train_X, train_y)
            
            y_pred_test = clf.predict(test_X) 
            y_pred_train = clf.predict(train_X)  
            

        else:
            return 'what model??'    


        res_test = np.mean(test_y == y_pred_test) 
        res_train = np.mean(train_y == y_pred_train)    

        ac_test = round(np.mean(res_test),4)
        ac_train = round(np.mean(res_train),4)             
        acs.append([ac_train,ac_test])     
               
        k+=1
      
    r_train = round(np.mean(np.array(acs)[:,0]),3)
    r_test = round(np.mean(np.array(acs)[:,1]),3)
       
    print('Mean train accuracy:', r_train)
    #print(np.array(acs)[:,0])
    print('Mean test accuracy:', r_test)
    #print(np.array(acs)[:,1])
    print(f'time to compute:', datetime.now() - startTime) 
    
    #return np.array(acs)
    return np.mean(np.array(acs)[:,1])


def run_batch():

    res = {}
    for decoder in ['LR','LDA']:
        y, regs, labs_, emb = get_data()
        lbs = {'labs': labs_, 'regs': regs}    

        # decode regs, leba using all cells    
        r = {}
        for lb in lbs:               
            ac = decode(y,lbs[lb],decoder=decoder)
            r[f'{lb}_all_n'] = ac
            r[f'{lb}_all_shuf'] = []
            for i in range(150):
                r[f'{lb}_all_shuf'].append(decode(y,lbs[lb],
                                           decoder=decoder,
                                           shuf=True))
                print(decoder, 'all', i, 'of', 150)
                                          
        res[f'all_{decoder}'] = r 
                               
        r = {}                                   
        # decode labs per region
        regs_ = list(Counter(regs))
        
        for reg in regs_:

            ac = decode(y[regs == reg],labs_[regs == reg],decoder=decoder)
            r[f'labs_{reg}_n'] = ac
            r[f'labs_{reg}_shuf'] = []
            d = deepcopy(labs_[regs == reg])
            for i in range(150):
                r[f'labs_{reg}_shuf'].append(decode(y[regs == reg],
                                           d, decoder=decoder,
                                           shuf=True))
                print(decoder, reg, i, 'of', 150)
                                       
        res[f'per_reg_{decoder}'] = r
                                  
                                     
    np.save(f'/home/mic/repro_decode.npy',res, allow_pickle=True)            
        

def plot_violin(metric = 'm'):
    
    '''
    plotting results from batch_job() as violins
    
    variable  # domain or vintage
    train  # display training accuracy
    '''
    
    R = np.load('/home/mic/repro_decode.npy', 
                allow_pickle=True).flat[0]
    
    fig = plt.figure(figsize=(6,3))
    ax = plt.subplot(1,2,2)  

    columns=['shuf', 'range', 'decoder', 'target', 'acc']        
    
    r = []
    for shuf in ['shuf','n']:
        for decoder in ['LDA', 'LR']:
            for target in ['labs', 'regs']:
                for ra in ['all', 'per_reg']:

                    if ra == 'all':
                        ac = R[f'{ra}_{decoder}'][f'{target}_all_{shuf}']
                        
                        r.append([shuf, ra, 
                                 decoder, target, ac])
                    if ra == 'per_reg':
                        if target == 'regs':
                            continue
                        regs = np.unique([x.split('_')[1] 
                            for x in R[f'per_reg_{decoder}'].keys()]) 
                    
                        for reg in regs:
                            ac = R[f'{ra}_{decoder}'][f'labs_{reg}_{shuf}']
                            
                            r.append([shuf, reg, 
                                     decoder, target, ac])                        

    df  = pd.DataFrame(data=r,columns=columns) 
    
    P = {}
    for decoder in ['LDA', 'LR']:
        for target in ['labs', 'regs']:
            ras = np.unique(df[np.bitwise_and(df['target'] == target,
                              df['decoder'] == decoder)]['range'].values)
            for ra in ras:
                null_d = df[np.bitwise_and.reduce([df['target'] == target,
                                       df['decoder'] == decoder,
                                       df['range'] == ra,
                                       df['shuf'] == 'shuf'])]['acc'].values[0]
                score =  df[np.bitwise_and.reduce([df['target'] == target,
                                       df['decoder'] == decoder,
                                       df['range'] == ra,
                                       df['shuf'] == 'n'])]['acc'].values[0]    
                                                         
                p =  np.mean(np.array(list(null_d) + 
                                      [score]) 
                                      >= score)

                P[f'{decoder}_{target}_{ra}'] = p
                                  
    return P
        

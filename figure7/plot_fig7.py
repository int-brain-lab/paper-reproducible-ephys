from one.api import ONE

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
from scipy.signal import correlate

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch
from ipywidgets import interact
from sklearn.metrics import r2_score
#import umap
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import string
from itertools import combinations
from statsmodels.stats.multitest import multipletests
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import Ridge

T_BIN = 0.02  # time bin size in seconds
one = ONE()

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



def all_panels(rm_unre=True, align='motion',split='RT',xyz_res=False,
               re_rank=2, fdr=True):


    # load metainfo df, row per cell
    concat_df = pd.read_csv('/home/mic/int-brain-lab/paper-reproducible-ephys/'
                     'figure7/data/figure8/figure8_dataframe.csv')
                     
    # load PSTHs, one per cell                 
    all_frs = np.load('/home/mic/int-brain-lab/paper-reproducible-ephys/'
                      'figure7/data/figure8/figure8_data_split_rt.npy')
    
    # get colors and short lab names                   
    _,b,lab_cols = labs_maps()
    ts = 'fast|slow RT PSTH'
    
    # include has minimum number of clusters as being 3
    all_frs = all_frs[concat_df['include'], :]
    concat_df = concat_df[concat_df['include']].reset_index()

    if rm_unre:
        # restrict to responsive units
        all_frs = all_frs[concat_df['responsive']]
        concat_df = concat_df[concat_df["responsive"]]


    y = all_frs
    regs = concat_df['region'].values
    labs = concat_df['lab'].values
    xyz = np.array([concat_df[g].values for g in ['x','y','z']]).T

    # exclude these labs
    lab3 = ['wittenlab','zadorlab','cortexlab','churchlandlab_ucla']
    
    bad = []
    for i in range(len(y)):                
        if labs[i] in lab3:
            bad.append(i)  
                          
    y = np.delete(y,bad,axis=0)
    regs = np.delete(regs,bad,axis=0)
    labs = np.delete(labs,bad,axis=0)  
    xyz = np.delete(xyz,bad,axis=0)                    

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
        

    # get reproduction using serveral first PCs
    u, s, vh = np.linalg.svd(y)
    y_res = {}
    
    for re_rank in [1,2,3]:#,30,100]:
        S_star = np.zeros(y.shape)
        for i in range(re_rank):
            S_star[i, i] = s[i]
        y_res[re_rank] = np.array(np.matrix(u) * np.matrix(S_star) * np.matrix(vh))
 
    xs = np.arange(len(y[0]))*T_BIN   

    # merge hofer and mirsicflogel labs
    labs[labs=='mrsicflogellab'] = 'hoferlab'

    
    fig = plt.figure(figsize=(10,10))



    
    inner = [['Ea'],
             ['Eb']]


    mosaic = [[inner,'Ga','G'],
              ['F','Ha','H'],
              ['B','Ia','I'],
              ['D','Ja','J'],
              ['L','Ka','K']]


#    mosaic = [['C','A','Ia','I','R'],
#              ['Ga','G','Ja','J','Cr'],
#              ['Ha','H','Ka','K','Cl'],
#              ['D','B','L',inner,'F']]
   
    ms2 = ['Ga','Ha','Ia','Ja','Ka']
    #panel_n = {
    
   
    axs = fig.subplot_mosaic(mosaic)
    
    cols_lab = [lab_cols[b[x]] for x in labs]                       
#    axs['A'].scatter(emb[:,0],emb[:,1],marker='o',c=cols_lab, s=2)

    labs_ = Counter(labs)

#    cents = {}
#    for lab in labs_:
#        cents[lab] = np.mean(emb[labs == lab],axis=0)

#    # plot means
#    for lab in cents:
#        axs['A'].scatter(cents[lab][0],cents[lab][1],
#                      s=800,marker='x',color=lab_cols[b[lab]])      

#    # plot confidence ellipses
#    for lab in labs_:
#        x = emb[labs == lab]
#        confidence_ellipse(x[:,0], x[:,1], axs['A'], n_std=1.0,
#                           edgecolor=lab_cols[b[lab]])

    # Euclidean distance of points for permutation test
    def distE(x,y):    
        return np.sqrt(np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y))    
     


    le = [Patch(facecolor=lab_cols[b[lab]], 
           edgecolor=lab_cols[b[lab]],
           label=b[lab]) for lab in labs_] 
#               
    axs['G'].legend(handles=le,
     loc='lower right', ncol=1, prop={'size': 6}).set_draggable(True)
#           
#    axs['A'].set_title('all cells')
#    axs['A'].set_xlabel('embedding dim 1')
#    axs['A'].set_ylabel('embedding dim 2')
#    axs['A'].text(-0.1, 1.30, 'A', transform=axs['A'].transAxes,
#      fontsize=16,  va='top', ha='right', weight='bold')
            
    
#    # plot average PSTHs across labs
#    for lab in labs_:     
#        ms = np.mean((y[labs == lab])/T_BIN,axis=0)
#        ste = np.std((y[labs == lab])/T_BIN,axis=0)/np.sqrt(len(y[labs == lab]))
#        
#        axs['C'].plot(xs,ms,color=lab_cols[b[lab]])
#        axs['C'].fill_between(xs, ms + ste, ms - ste, 
#                              color=lab_cols[b[lab]],alpha=0.2)
#       
#    axs['C'].set_title(ts)
#    axs['C'].set_xlabel('time [sec]')
#    axs['C'].set_ylabel('firing rate [Hz]')    
#    for x in np.array([25, 100])*T_BIN:
#        axs['C'].axvline(x=x, lw=0.5, linestyle='--', 
#                               c='g',label='motion start')
#                               
#    axs['C'].axvline(x=75*T_BIN, lw=2, linestyle='-', 
#                           c='cyan',label='trial cut')                               
#                               
#    axs['C'].text(-0.1, 1.30, 'C', transform=axs['C'].transAxes,
#      fontsize=16,  va='top', ha='right', weight='bold')
#    
#    le = [Line2D([0], [0], color='g', lw=0.5, ls='--', label='motion start'),
#          Line2D([0], [0], color='cyan', lw=2, ls='-', label='trial cut')] 
#               
#    axs['C'].legend(handles=le,
#     loc='lower right', ncol=1, prop={'size': 6}).set_draggable(True)  
                                    
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
     loc='lower right', ncol=1, prop={'size': 6}).set_draggable(True) 
          
    axs['B'].set_title('all cells')
    axs['B'].set_xlabel('embedding dim 1')
    axs['B'].set_ylabel('embedding dim 2')
    axs['B'].text(-0.1, 1.30, 'B', transform=axs['B'].transAxes,
      fontsize=16,  va='top', ha='right', weight='bold')
    
    axs['B'].sharex(axs['H'])
    axs['B'].sharey(axs['H']) 
    axs['D'].sharex(axs['Ga'])
    axs['D'].sharey(axs['Ga']) 
    
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
    axs['D'].text(-0.1, 1.30, 'D', transform=axs['D'].transAxes,
      fontsize=16,  va='top', ha='right', weight='bold')
      
    le = [Line2D([0], [0], color='g', lw=0.5, ls='--', label='motion start'),
          Line2D([0], [0], color='cyan', lw=2, ls='-', label='trial cut')] 
               
    axs['D'].legend(handles=le,
     loc='lower right', ncol=1, prop={'size': 6}).set_draggable(True)
  
    '''
    check mosaic plot function to get different space allocation
    '''

    # illustrate example cells
    # 1 with great, 1 with bad reconstruction     
    r2s_ = {}     
    for re_rank in y_res:
        r2s = []
        iex = []
        for i in range(len(y)):
            r2s.append(r2_score(y[i],y_res[re_rank][i]))
            iex.append(i)            
   
        r2s_[re_rank] = r2s
        if re_rank == 2:
            t = np.argsort(r2s)   
            t_ = np.concatenate([t[170:171],t[-1:]])#,t[len(t)//2:len(t)//2 +1]
            idxs = [iex[j] for j in t_]
        
        
    xs = np.arange(len(y[0]))*T_BIN
    
    # split good and bad example cell into two panels
    ms = ['Ea','Eb']
       
    for k in range(len(idxs)):

        axs[ms[k]].plot(xs,y[idxs[k]]/T_BIN,c='k',label='PSTH')
        axs[ms[k]].plot(xs,y_res[2][idxs[k]]/T_BIN,c='r',ls='--',label='fit')

        for x in np.array([25, 100])*T_BIN:
            axs[ms[k]].axvline(x=x, linewidth=0.5, linestyle='--', 
                                   c='g',label='motion start')
        axs[ms[k]].axvline(x=75*T_BIN, lw=2, linestyle='-', 
                               c='cyan',label='trial cut')               
        axs[ms[k]].set_ylabel('firing rate \n [Hz]')
        axs[ms[k]].text(1, 1.1, 
            rf'$r^2$={np.round(r2_score(y[idxs[k]],y_res[2][idxs[k]]),2)}',
            transform=axs[ms[k]].transAxes,
            fontsize=8,  va='top', ha='right') 
        k+=1                       
        
        
    axs['Eb'].set_xlabel('time [sec]')
    
    axs['Ea'].set_title('2-PCs reconstruction')
        
    axs['Ea'].text(-0.1, 2.4, 'E', transform=axs['Ea'].transAxes,
      fontsize=16,  va='top', ha='right', weight='bold')
      
      
    le = [Line2D([0], [0], c='r',ls='--', label='fit'),
          Line2D([0], [0], c='k', label='PSTH')] 
               
    axs['Ea'].legend(handles=le,
     loc='lower right', ncol=1, prop={'size': 6}).set_draggable(True)
     
     
    for re_rank in r2s_:
        _,patches,_ =axs['F'].hist(x=r2s_[re_rank], bins='auto',label=re_rank,
                alpha=0.7, rwidth=0.85,
                histtype=u'step',lw=2)

    leg = axs['F'].legend(ncol=len(r2s_[re_rank])//4, 
                    prop={'size': 6}).set_draggable(True)
    axs['F'].set_xlabel(r'$r^2$')
    axs['F'].set_ylabel('number of neurons')
    axs['F'].set_title(f'goodness of PSTH \n reconstruction')     
    axs['F'].text(-0.1, 1.30, 'F', transform=axs['F'].transAxes,
      fontsize=16,  va='top', ha='right', weight='bold')
    
    # per region dim red
    ms = ['G','H','I','J','K']
    ms2 = ['Ga','Ha','Ia','Ja','Ka']
    
    k = 0
    
    p_r = {}
    D = Counter(regs)
    for reg in D:
        if reg is None:
            continue
        
        y2 = y[regs == reg]
        labs2 = labs[regs == reg]

        emb2 = emb[regs == reg]


        cols_lab = [lab_cols[b[x]] for x in labs2]                      
        axs[ms[k]].scatter(emb2[:,0],emb2[:,1],marker='o',c=cols_lab, s=2)

        labs_ = Counter(labs2)

        cents = {}
        for lab in labs_:
            cents[lab] = np.mean(emb2[labs2 == lab],axis=0)

#        # plot means
        for lab in cents:
            axs[ms[k]].scatter(cents[lab][0],cents[lab][1],
                          s=800,marker='x',color=lab_cols[b[lab]])      

        # plot confidence ellipses
        for lab in labs_:
            x = emb2[labs2 == lab]
            confidence_ellipse(x[:,0], x[:,1], axs[ms[k]], n_std=1.0,
                               edgecolor=lab_cols[b[lab]])

        # shuffle test
        nrand = 100  #random lab allocations    
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
        axs[ms[k]].text(-0.1, 1.30, ms[k], transform=axs[ms[k]].transAxes,
          fontsize=16,  va='top', ha='right', weight='bold')
          
        axs[ms[k]].sharex(axs['B'])
        axs[ms[k]].sharey(axs['B']) 
        
        # plot average PSTHs across labs
        for lab in labs_:  
   
            mes = np.mean((y2[labs2 == lab])/T_BIN,axis=0)
            ste = np.std((y2[labs2 == lab])/T_BIN,axis=0)/np.sqrt(
                                                len(y2[labs2 == lab]))

            
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
                                   
        axs[ms2[k]].text(-0.1, 1.30, ms2[k], transform=axs[ms2[k]].transAxes,
          fontsize=16,  va='top', ha='right', weight='bold')
        
        le = [Line2D([0], [0], color='g', lw=0.5, ls='--', label='motion start'),
              Line2D([0], [0], color='cyan', lw=2, ls='-', label='trial cut')] 

#        if k == 0:                   
#            axs[ms2[k]].legend(handles=le,
#             loc='lower right', ncol=1).set_draggable(True)         

        axs[ms2[k]].sharex(axs['D'])
        axs[ms2[k]].sharey(axs['D'])
                
        k+=1

    # plot permutation test p values for regional scatters 
    a = np.zeros((len(p_r),len(p_r[list(p_r.keys())[0]])))
    # multiple comparison correction
    pvals = [p_r[reg][lab] for  reg in p_r for lab in p_r[reg]]
    _, pvals_c, _, _ = multipletests(pvals, 0.05, method='fdr_bh')
    
    if fdr:
        pps = pvals_c
    else: 
        pps = pvals
    
    p_rc = {}
    i = 0
    for reg in p_r:
        p_rc[reg] = {}
        for lab in p_r[reg]:
            p_rc[reg][lab] = np.round(pps[i],3)        
            i += 1
   
    i = 0
    for reg in p_rc:
        j = 0
        for pair in p_rc[reg]:
            #if p_r[reg][pair] < 0.05:
            a[i,j] = p_rc[reg][pair]
            j += 1                        
        i += 1
        
    im = axs['L'].imshow(a, cmap='cool',aspect="auto",#'Greys'
        interpolation='none')
    # print values onto image
    for (i,j), s in np.ndenumerate(a):
        axs['L'].text(j,i,s,ha='center',va='center', fontsize=8)         
            
    cb = fig.colorbar(im, ax=axs['L'], location='right', anchor=(0, 0.3), shrink=0.7)
    cb.ax.set_ylabel('p')
    axs['L'].set_xticks(range(len(p_r[reg].keys())))
    axs['L'].set_xticklabels([b[lab] for lab in p_r[reg].keys()],rotation = 90)
    axs['L'].set_yticks(range(len(p_r.keys())))
    axs['L'].set_yticklabels(p_r.keys())    
    
    axs['L'].set_title(f'permutation test')
    #axs['L'].set_xlabel(f'labs')
    #axs['L'].set_ylabel('regions') 
    axs['L'].text(-0.1, 1.30, 'L', transform=axs['L'].transAxes,
      fontsize=16,  va='top', ha='right', weight='bold')


#    # reaction time histograms
#    rts = np.load('/home/mic/repro_manifold/rts.npy',
#                  allow_pickle=True).flat[0]


#    for lab in rts:
#        if lab in ['UCLA']:
#            continue  
#        rtsf = np.array(rts[lab])
#        axs['R'].hist(rtsf[np.where(rtsf<5)[0]],bins=200,
#                      label=lab,histtype=u'step',lw=2)
#    axs['R'].set_xlabel('reaction time [sec]')
#    axs['R'].set_ylabel('frequency')
#    #axs['R'].set_title('pooled reaction times \n of repeated site sessions')
#    axs['R'].legend().set_draggable(True)
#    axs['R'].set_xlim([0, 0.5])    
#    axs['R'].text(-0.1, 1.30, 'R', transform=axs['R'].transAxes,
#      fontsize=16,  va='top', ha='right', weight='bold') 

#    
    #return emb, xyz, labs, regs    


    # shuffle test
#    random.shuffle(labs)
#    random.shuffle(regs)



#    vmin, vmax = np.amin([al, ar]), np.amax([al, ar])
#            
#    dat = {'Cr':ar, 'Cl':al}
#    met = {'Cr':'regions', 'Cl':'labs'}
#    xlabels = [s1+','+s2 for s1 in ['emb1','emb2'] for s2 in ['x','y','z']]
#    ylabels = {'Cr':regs_, 'Cl':[b[l] for l in labs_]}
#                
#    for p in ['Cr','Cl']:
#        
#        im = axs[p].imshow(dat[p], cmap='cool',aspect="auto",#'Greys'
#                           interpolation='none',vmin=vmin, vmax=vmax)      
#        cb = fig.colorbar(im, ax=axs[p], location='right', 
#                          anchor=(0, 0.3), shrink=0.7)
#        cb.ax.set_ylabel('max(cc)')
#        axs[p].set_xticks(range(dat[p].shape[1]))
#        axs[p].set_xticklabels(xlabels,rotation = 90)
#        axs[p].set_yticks(range(dat[p].shape[0]))
#        axs[p].set_yticklabels(ylabels[p])    
#        
#        axs[p].set_title(f'max cross correlation')
#        axs[p].set_xlabel(f'covariate pairs')
#        axs[p].set_ylabel(met[p]) 
#        axs[p].text(-0.1, 1.30, p, transform=axs[p].transAxes,
#          fontsize=16,  va='top', ha='right', weight='bold')


    supt = (f'PCA dim reduction of PSTH; {len(y)} clusters; xyz_res = {xyz_res}'
             f'; responsive only {rm_unre};'
             f' fdr = {fdr}; align = {align}; split = {split}')       
    plt.suptitle(supt)     
    plt.tight_layout()
    


    
    
        




def plot_reaction_time_hists(rts=None,pool=False):

    #rts = np.load('/home/mic/repro_manifold/rts.npy',allow_pickle=True).flat[0]
    _,bl,lab_cols = labs_maps()
    fig = plt.figure()
    lab3 = ['wittenlab','zadorlab','cortexlab','churchlandlab_ucla']
    if rts is None:
        inserts = get_repeated_sites()
        eids = np.array(inserts)[:,0]
        
        rts = {}
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
                try:
                    rts[bl[lab]].append(a-b)
                except:
                    rts[bl[lab]] = []
                    rts[bl[lab]].append(a-b)       
             
    for lab in rts:    
        rtsf = np.array(rts[lab])
        plt.hist(rtsf[np.where(rtsf<5)[0]],bins=200,label=lab)
    plt.xlabel('reaction time [sec]')
    plt.ylabel('frequency')
    #plt.title('reaction times')
    np.save('/home/mic/repro_manifold/rts.npy',rts,allow_pickle=True)
    return rts




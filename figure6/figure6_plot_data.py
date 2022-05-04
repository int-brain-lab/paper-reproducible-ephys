import numpy as np
from collections import Counter

from scipy.stats import percentileofscore
import random

from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import r2_score

from matplotlib.lines import Line2D

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import string

from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import Ridge

from reproducible_ephys_functions import filter_recordings, save_figure_path, labs, figure_style
from figure6.figure6_load_data import load_dataframe, load_data

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


def all_panels(rm_unre=True, align='move', split='rt', xyz_res=False, re_rank=2, fdr=True, permute_include=True):

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
        concat_df = concat_df[concat_df["responsive"]]

    y = all_frs
    regs = concat_df['region'].values
    labs = concat_df['lab'].values
    xyz = np.array([concat_df[g].values for g in ['x', 'y', 'z']]).T

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

    # merge hofer and mirsicflogel labs
    labs[labs == 'mrsicflogellab'] = 'hoferlab'

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
    figs = plt.figure(figsize=(8, 9), facecolor='w')

    inner = [['Ea'],
             ['Eb']]

    mosaic = [[inner, 'F'],
              ['B', 'D'],
              ['G', 'Ga']]

    mosaic_supp = [['Ha', 'H'],
                   ['Ia', 'I'],
                   ['Ja', 'J'],
                   ['Ka', 'K']]

    mf = np.array(mosaic, dtype=object).flatten()
    mf[0] = 'Ea'
    panel_n = dict(zip(mf, string.ascii_lowercase))

    mfs = np.array(mosaic_supp, dtype=object).flatten()
    panel_ns = dict(zip(mfs, string.ascii_lowercase))

    axs = fig.subplot_mosaic(mosaic)
    axss = figs.subplot_mosaic(mosaic_supp)

    labs_ = Counter(labs)
    # Euclidean distance of points for permutation test
    #return labs_
    def distE(x, y):
        return np.sqrt(np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y))

    le_labs = [Patch(facecolor=lab_cols[b[lab]], edgecolor=lab_cols[b[lab]], label=b[lab]) for lab in labs_]

    axs['G'].legend(handles=le_labs, loc='lower left', bbox_to_anchor=(0.1, 1), ncol=3, prop={'size': 8}).set_draggable(True)

    Dc = figure_style(return_colors=True)

    regs_ = Counter(regs)
    cols_reg = [Dc[x] for x in regs]
    axs['B'].scatter(emb[:, 0], emb[:, 1], marker='o', c=cols_reg, s=2)

    regs_ = Counter(regs)
    cents = {}
    for reg in regs_:
        cents[reg] = np.mean(emb[regs == reg], axis=0)

    # shuffle test
    nrand = 1000  # random region allocations
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

    p = 1 - (0.01 * percentileofscore(null_d, dist))
    p = np.round(p, )

    print('region permutation test:', p)

    # plot means
    for reg in cents:
        axs['B'].scatter(cents[reg][0], cents[reg][1], s=800, marker='x', color=Dc[reg])

    # plot confidence ellipses
    for reg in regs_:
        x = emb[regs == reg]
        confidence_ellipse(x[:, 0], x[:, 1], axs['B'], n_std=1.0, edgecolor=Dc[reg])

    le = [Patch(facecolor=Dc[reg], edgecolor=Dc[reg], label=reg) for reg in regs_]

    axs['B'].legend(handles=le, bbox_to_anchor=(0.3, 1), loc='lower left', ncol=3, prop={'size': 7}).set_draggable(True)

    axs['B'].set_title('all cells', loc='left')
    axs['B'].set_xlabel('embedding dim 1')
    axs['B'].set_ylabel('embedding dim 2')
    axs['B'].text(-0.1, 1.30, panel_n['B'], transform=axs['B'].transAxes, fontsize=16, va='top', ha='right', weight='bold')

    axs['B'].sharex(axss['H'])
    axs['B'].sharey(axss['H'])
    axs['D'].sharex(axs['Ga'])
    axs['D'].sharey(axs['Ga'])

    # plot average PSTHs
    for reg in regs_:

        labsinreg = len(Counter(labs[regs == reg]))
        sessinreg = len(Counter(sess[regs == reg]))

        ms = np.mean((y[regs == reg]) / T_BIN, axis=0)
        ste = np.std((y[regs == reg]) / T_BIN, axis=0) / np.sqrt(sessinreg)

        # only plot one PSTH
        ms = ms[:len(xs)]
        ste = ste[:len(xs)]

        axs['D'].plot(xs - 0.5, ms, color=Dc[reg])
        axs['D'].fill_between(xs - 0.5, ms + ste, ms - ste, color=Dc[reg], alpha=0.2)

    for x in np.array([25]) * T_BIN:
        axs['D'].axvline(x=0, linewidth=0.5, linestyle='--', c='g', label='motion start')

    axs['D'].set_xlabel('time [sec]')
    axs['D'].set_ylabel('firing rate [Hz]')
    axs['D'].text(-0.1, 1.30, panel_n['D'], transform=axs['D'].transAxes, fontsize=16, va='top', ha='right', weight='bold')

    le = [Line2D([0], [0], color='g', lw=0.5, ls='--', label='motion start')]

    axs['D'].legend(handles=le, bbox_to_anchor=(0.3, 1), loc='lower left', ncol=1, prop={'size': 7}).set_draggable(True)

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
            t_ = np.concatenate([t[170:171], t[-4:-3]])
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
                               c='g', label='motion start')

        axs[ms[k]].set_ylabel('firing rate \n [Hz]')
        stext = rf'$r^2$={np.round(r2_score(y[idxs[k]], y_res[2][idxs[k]]), 2)}'
        axs[ms[k]].text(0.2, 1, stext, transform=axs[ms[k]].transAxes,
                        fontsize=7, va='top', ha='right')

        if k == 1:
            axs[ms[k]].set_xlabel('time [sec]')

        if k == 0:
            axs[ms[k]].text(-0.1, 1.6, panel_n[ms[k]],
                            transform=axs[ms[k]].transAxes,
                            fontsize=16, va='top',
                            ha='right', weight='bold')
            axs[ms[k]].legend(loc='lower left', bbox_to_anchor=(0, 0.9), ncol=3, prop={'size': 7}).set_draggable(True)
        k += 1

    for re_rank in r2s_:
        _, patches, _ = axs['F'].hist(x=r2s_[re_rank], bins='auto', label=re_rank, alpha=0.7, rwidth=0.85, histtype=u'step', lw=2)

    leg = axs['F'].legend(ncol=len(r2s_[re_rank]) / 4, prop={'size': 7}).set_draggable(True)
    axs['F'].set_xlabel(r'$r^2$')
    axs['F'].set_ylabel('number of neurons')
    axs['F'].text(-0.1, 1.30, panel_n['F'], transform=axs['F'].transAxes, fontsize=16, va='top', ha='right', weight='bold')

    # per region dim red
    ms = ['G', 'H', 'I', 'J', 'K']
    ms2 = ['Ga', 'Ha', 'Ia', 'Ja', 'Ka']

    k = 0
    p_ = {}  # Guido's permutation test score
    p_r = {}
    D = sorted(list(Counter(regs)))
    
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
        axs3[ms[k]].scatter(emb2[:, 0], emb2[:, 1], marker='o', c=cols_lab, s=2)

        labs_ = Counter(labs2)

        cents = {}
        for lab in labs_:
            cents[lab] = np.mean(emb2[labs2 == lab], axis=0)

        # plot means
        for lab in cents:
            axs3[ms[k]].scatter(cents[lab][0], cents[lab][1], s=800, marker='x', color=lab_cols[b[lab]])

        # plot confidence ellipses
        for lab in labs_:
            x = emb2[labs2 == lab]
            confidence_ellipse(x[:, 0], x[:, 1], axs3[ms[k]], n_std=1.0, edgecolor=lab_cols[b[lab]])

        # shuffle test
        centsr = []
        for shuf in range(nrand):
            labsr = labs2.copy()
            random.shuffle(labsr)
            cenr = {}
            for lab in labs_:
                cenr[lab] = np.mean(emb2[labsr == lab], axis=0)
            centsr.append(cenr)

        ps = {}
        for lab in cents:
            cs = np.mean([cents[l] for l in cents], axis=0)  # if l!=lab
            dist = distE(cents[lab], cs)
            null_d = [distE(cenr[lab],
                      np.mean([cenr[l] for l in cenr], axis=0))  # if l!=lab
                      for cenr in centsr]
            p = 1 - (0.01 * percentileofscore(null_d, dist))
            ps[lab] = np.round(p, 3)
        p_r[reg] = ps

        # Guido's permutation test
        cs = np.mean([cents[lab] for lab in cents], axis=0)
        dist = sum([distE(cents[lab], cs) for lab in cents])

        null_d = [sum([distE(cenr[lab], np.mean([cenr[la] for la in cenr], axis=0)) for lab in cenr]) for cenr in centsr]
        p = 1 - (0.01 * percentileofscore(null_d, dist))
        p_[reg] = np.round(p, 3)

        axs3[ms[k]].set_title(reg, loc='left')
        axs3[ms[k]].set_xlabel('embedding dim 1')
        axs3[ms[k]].set_ylabel('embedding dim 2')
        axs3[ms[k]].text(-0.1, 1.30, panel_n3[ms[k]],
                         transform=axs3[ms[k]].transAxes,
                         fontsize=16, va='top',
                         ha='right', weight='bold')

        if k == 1:
            axs3[ms[k]].legend(handles=le_labs, loc='lower left', bbox_to_anchor=(0.1, 1), ncol=3,
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
            axs3[ms2[k]].fill_between(xs - 0.5, mes + ste, mes - ste, color=lab_cols[b[lab]], alpha=0.2)

        axs3[ms2[k]].set_title(reg)
        axs3[ms2[k]].set_xlabel('time [sec]')
        axs3[ms2[k]].set_ylabel('firing rate [Hz]')
        for x in np.array([25]) * T_BIN:
            axs3[ms2[k]].axvline(x=0, lw=0.5, linestyle='--', c='g', label='motion start')

        axs3[ms2[k]].text(-0.1, 1.30, panel_n3[ms2[k]],
                          transform=axs3[ms2[k]].transAxes, fontsize=16, va='top', ha='right', weight='bold')

        le = [Line2D([0], [0], color='g', lw=0.5, ls='--', label='motion start')]

        axs3[ms[k]].sharex(axs['B'])
        axs3[ms[k]].sharey(axs['B'])
        axs3[ms2[k]].sharex(axs['D'])
        axs3[ms2[k]].sharey(axs['D'])

        k += 1

    # plot permutation test p values for regional scatters
    a = np.zeros((len(p_r), len(p_r[list(p_r.keys())[0]])))
    # multiple comparison correction
    pvals = [p_r[reg][lab] for reg in p_r for lab in p_r[reg]]
    _, pvals_c, _, _ = multipletests(pvals, 0.05, method='fdr_bh')

    pvals_ = [p_[reg] for reg in p_]
    _, pvals_c_, _, _ = multipletests(pvals_, 0.05, method='fdr_bh')

    print('permutation test labs per region, corrected:')
    print(dict(zip(list(p_.keys()), [np.round(p, 3) for p in pvals_c_])))

    axs['B'].set_xlim([-1.3, 1])
    axs['B'].set_ylim([-0.7, 0.7])
    axs['D'].set_xlim([-0.2, 0.8])
    axs['D'].set_ylim([-3.1, 8.5])

    fig.tight_layout()
    figs.tight_layout()

    fig_path = save_figure_path(figure='figure6')
    print(f'Saving figures to {fig_path}')
    fig.savefig(fig_path.joinpath('figure6.png'))
    figs.savefig(fig_path.joinpath('figure6_supp1.png'))

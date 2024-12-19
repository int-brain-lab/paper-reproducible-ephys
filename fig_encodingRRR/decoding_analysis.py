from utils.decoding_analysis import make_CV_prediction, trainpred_func_SVC
from reproducible_ephys_functions import (figure_style, get_row_coord, get_label_pos, plot_horizontal_institute_legend,
                                          BRAIN_REGIONS, REGION_RENAME)
from utils.utils import make_folder, log_kv

import pdb, os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import figrid as fg

def plot_scatter_res(X, y, yi2label, laborarea="area",ax=None, legend=True):
    n_clus = len(np.unique(y))
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(4,3.5))
    pca = PCA(n_components=2)
    x_lowd = pca.fit_transform(X)
    for yi in np.unique(y):
        if laborarea == "area":
            ax.scatter(*x_lowd[y==yi].T, c=region_colors[yi2label[yi]], # c=plt.get_cmap('tab10')(yi), 
                        s=1, label=REGION_RENAME[yi2label[yi]], alpha=0.3)
        else:
            ax.scatter(*x_lowd[y==yi].T, c=plt.get_cmap('tab10')(yi), 
                        s=1, label=yi2label[yi], alpha=0.3)
    ax.set_xlabel("Dim1")
    ax.set_ylabel("Dim2")
    if legend:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                fancybox=True, shadow=True, ncol=3)
    plt.tight_layout()

def dec_perf(data, labels1, labels2, print_it=False, plot_it=False):
    res = make_CV_prediction(data, labels1, trainpred_func_SVC, eid_all=labels2)
    return res['perf']['f1']


def main_clean(X, y, eids, n_permut, stest_fname, 
               plot_kwargs):
    res = make_CV_prediction(X, y, trainpred_func_SVC, eid_all=eids)
    f1 = res['perf']['f1']; 
    _f = os.path.join(resgood_folder, stest_fname)
    if os.path.isfile(_f):
        null_dist = np.load(_f)
        p = len(null_dist[null_dist > f1]) / len(null_dist)
    else:
        from permutation_test import  permut_test
        p, _, null_dist = permut_test(X, dec_perf, y, eids, 
                                    n_permut=n_permut, 
                                    shuffling='labels1' if plot_kwargs['laborarea'] == "area" else 'labels1_based_on_2', 
                                    return_details=True, n_cores=1)
        np.save(_f, np.asarray(null_dist))
    log_kv(f1=f1, p=p)
    # for visualization
    axes = plot_kwargs['axes']
    clf = LinearDiscriminantAnalysis(n_components=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    sel_areas_lowd = clf.fit(X_train, y_train).transform(X_test)
    ax = axes[1]
    plot_scatter_res(sel_areas_lowd, y_test, plot_kwargs['labels'], 
                 ax=ax, legend=False, laborarea=plot_kwargs['laborarea'])
    
    ax = axes[0]
    sns.histplot(null_dist, ax=ax, stat="count", label='permuted', binwidth=0.01)
    ax.axvline(x=f1, color='r', linestyle='--', label='original')
    ax.set_xlabel("Macro F1")
    ax.set_ylabel("Number of \n permutations")
    if ('legend' in plot_kwargs) and (plot_kwargs['legend']): 
        ax.legend(fontsize='small', frameon=True, 
                    loc='upper right', bbox_to_anchor=(1.5, 1.0))
    # ax.set_title(f"{plot_kwargs['title']}")
    ax.set_title(f"{plot_kwargs['title']} \n p: {p:.3f}")

    return res


### load data
resgood_folder = make_folder("./result")
data = np.load(os.path.join(resgood_folder, "data.npz"), allow_pickle=True)
nis_incmask = (data[f"r2s"] >= 0.03)
coef_vs = data['coef_vs'][nis_incmask]
eids = data['eids'][nis_incmask]
acronyms = data['acronyms'][nis_incmask]
labs = data['labs'][nis_incmask]

eid_list = np.unique(eids)
eid2li = {a: ai for ai, a in enumerate(eid_list)}
area_list = ['VISa_am', 'CA1','DG','LP','PO']
area2ai = {a: ai for ai, a in enumerate(area_list)}
ai2area = {ai: "PPC" if a == "VISa_am" else a for ai, a in enumerate(area_list)}
lab_list = np.unique(labs)
lab2li = {a: ai for ai, a in enumerate(lab_list)}
li2lab = {ai:a for ai, a in enumerate(lab_list)}


### prepare figure
region_colors = figure_style(return_colors=True)
width = 9
height = 4
fig = plt.figure(figsize=(width, height))
padx=0.5
xspans = get_row_coord(width, [1, 5, 5, 5, 5, 5, 5, 5], hspace=[0.5, 0.6, 0.5, 0.3, 0.3, 0.3, 0.3], pad=padx)
xspans_labels = get_row_coord(width, [1, 5, 25], hspace=[0.5, 0.6])
yspans = get_row_coord(height, [10, 10, 2], hspace=[0.6, 0.4], pad=0.3)


axs = {
    'label1': fg.place_axes_on_grid(fig, xspan=xspans[0], yspan=yspans[0]),
    'label2': fg.place_axes_on_grid(fig, xspan=xspans[0], yspan=yspans[1]),
    'regions': fg.place_axes_on_grid(fig, xspan=xspans_labels[1], yspan=yspans[2]),
    'labs': fg.place_axes_on_grid(fig, xspan=xspans_labels[2], yspan=yspans[2]),
    'A_1': fg.place_axes_on_grid(fig, xspan=xspans[1], yspan=yspans[0]),
    'A_2': fg.place_axes_on_grid(fig, xspan=xspans[1], yspan=yspans[1]),
    'B_1': fg.place_axes_on_grid(fig, xspan=xspans[2], yspan=yspans[0]),
    'B_2': fg.place_axes_on_grid(fig, xspan=xspans[2], yspan=yspans[1]),
    'B_VISa_am_1': fg.place_axes_on_grid(fig, xspan=xspans[3], yspan=yspans[0]),
    'B_VISa_am_2': fg.place_axes_on_grid(fig, xspan=xspans[3], yspan=yspans[1]),
    'B_CA1_1': fg.place_axes_on_grid(fig, xspan=xspans[4], yspan=yspans[0]),
    'B_CA1_2': fg.place_axes_on_grid(fig, xspan=xspans[4], yspan=yspans[1]),
    'B_DG_1': fg.place_axes_on_grid(fig, xspan=xspans[5], yspan=yspans[0]),
    'B_DG_2': fg.place_axes_on_grid(fig, xspan=xspans[5], yspan=yspans[1]),
    'B_LP_1': fg.place_axes_on_grid(fig, xspan=xspans[6], yspan=yspans[0]),
    'B_LP_2': fg.place_axes_on_grid(fig, xspan=xspans[6], yspan=yspans[1]),
    'B_PO_1': fg.place_axes_on_grid(fig, xspan=xspans[7], yspan=yspans[0]),
    'B_PO_2': fg.place_axes_on_grid(fig, xspan=xspans[7], yspan=yspans[1]),
}
labels = [{'label_text': 'a', 'xpos': get_label_pos(width, xspans[1][0], pad=padx),
           'ypos': get_label_pos(height, yspans[0][0], pad=0.3),
           'fontsize': 10, 'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
          {'label_text': 'b', 'xpos': get_label_pos(width, xspans[2][0], pad=padx),
           'ypos': get_label_pos(height, yspans[0][0], pad=0.3), 'fontsize': 10,
           'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
           {'label_text': 'c', 'xpos': get_label_pos(width, xspans[3][0], pad=padx),
           'ypos': get_label_pos(height, yspans[0][0], pad=0.3), 'fontsize': 10,
           'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
          ]
fg.add_labels(fig, labels)
adjust = 0.3
fig.subplots_adjust(top=1 - adjust / height, bottom=(adjust - 0.2) / height, left=(adjust) / width,
                     right=1 - (adjust) / width)
axs['label1'].set_axis_off()
axs['label1'].annotate('Decoding', xy=(0, 0.5), xycoords='axes fraction', size='8', ha='right', va='center', rotation='vertical')
axs['label2'].set_axis_off()
axs['label2'].annotate('LDA', xy=(0, 0.5), xycoords='axes fraction', size='8', ha='right', va='center', rotation='vertical')


### decoding analysis
n_permut=1000
for ci, k in enumerate(['acronym','lab']):
    if k == "lab":
        _ys = np.asarray([lab2li[l] for l in labs])
        labels = li2lab
        title = "All neurons"
        laborarea = 'lab'
        ax = [axs['B_1'], axs['B_2']]
    elif k == 'acronym':
        _ys = np.asarray([area2ai[l] for l in acronyms])
        labels = ai2area
        title = "All neurons"
        laborarea = 'area'
        ax = [axs['A_1'], axs['A_2']]
    eid_all = np.array([eid2li[a] for a in eids]).astype(int) 
    print(k)
    for _yi in np.unique(_ys):
        print(_yi, len(np.unique(eid_all[_ys==_yi])))
    ### evaluate the performance
    res = main_clean(coef_vs, _ys, eid_all, n_permut, f"{k}_decoding_{n_permut}.npy", 
                     plot_kwargs=dict(axes=ax, title=title, labels=labels, legend=False, laborarea=laborarea))

# per area lab decoding
for ai, area in enumerate(area_list):
    area_mask = acronyms==area
    _ys = np.asarray([lab2li[l] for l in labs[area_mask]])
    eid_all = np.array([eid2li[a] for a in eids[area_mask]]).astype(int) 
    print(area)
    for _yi in np.unique(_ys):
        print(_yi, len(np.unique(eid_all[_ys==_yi])))
    title = f"VISa/am neurons" if area == "VISa_am" else f"{area} neurons"
    labels = li2lab
    ax = [axs[f'B_{area}_{_+1}'] for _ in range(2)]
    ### evaluate the performance
    res = main_clean(coef_vs[area_mask], _ys, eid_all, n_permut, f"{area}_lab_decoding_{n_permut}.npy", 
                     plot_kwargs=dict(axes=ax, title=title, labels=labels, legend=area=='PO', laborarea='lab'))



# Remove some axis
for key, ax in axs.items():
    if key[0] != 'B':
        continue
    if key in ["B_1", "B_2"]: 
        continue
    if 'B_VISa' not in key:
        ax.set_ylabel('')

axs['regions'].set_axis_off()
for i, reg in enumerate(BRAIN_REGIONS):
    if i == 0:
        text = axs['regions'].text(-0.3, 0.5, REGION_RENAME[reg], color=region_colors[reg], fontsize=8,
                       transform=axs['regions'].transAxes)
    else:
        text = axs['regions'].annotate(
            '  ' + REGION_RENAME[reg], xycoords=text, xy=(1, 0), verticalalignment="bottom",
            color=region_colors[reg], fontsize=8)

axs['labs'].set_axis_off()
# Change this so the labs are roughly centered in the axis
offset = 0.05
plot_horizontal_institute_legend(lab_list, axs['labs'], offset=offset)


plt.savefig(os.path.join(resgood_folder, f"figure_7.pdf")); plt.close('all')


from reproducible_ephys_functions import (figure_style, get_row_coord, get_label_pos, plot_horizontal_institute_legend,
                                          BRAIN_REGIONS, REGION_RENAME)
import matplotlib.pyplot as plt
import figrid as fg

region_colors = figure_style(return_colors=True)
width = 7
height = 5
fig = plt.figure(figsize=(width, height))

xspans = get_row_coord(width, [1, 5, 5, 5, 5, 5, 5], hspace=[0.5, 0.6, 0.2, 0.2, 0.2, 0.2])
xspans_labels = get_row_coord(width, [1, 5, 25], hspace=[0.5, 0.6])
yspans = get_row_coord(height, [10, 10, 10, 2], hspace=[0.5, 0.5, 0.4], pad=0.3)


axs = {
    'label1': fg.place_axes_on_grid(fig, xspan=xspans[0], yspan=yspans[0]),
    'label2': fg.place_axes_on_grid(fig, xspan=xspans[0], yspan=yspans[1]),
    'label3': fg.place_axes_on_grid(fig, xspan=xspans[0], yspan=yspans[2]),
    'regions': fg.place_axes_on_grid(fig, xspan=xspans_labels[1], yspan=yspans[3]),
    'labs': fg.place_axes_on_grid(fig, xspan=xspans_labels[2], yspan=yspans[3]),
    'A_1': fg.place_axes_on_grid(fig, xspan=xspans[1], yspan=yspans[0]),
    'A_2': fg.place_axes_on_grid(fig, xspan=xspans[1], yspan=yspans[1]),
    'A_3': fg.place_axes_on_grid(fig, xspan=xspans[1], yspan=yspans[2]),
    'B_VIs_1': fg.place_axes_on_grid(fig, xspan=xspans[2], yspan=yspans[0]),
    'B_VIs_2': fg.place_axes_on_grid(fig, xspan=xspans[2], yspan=yspans[1]),
    'B_VIs_3': fg.place_axes_on_grid(fig, xspan=xspans[2], yspan=yspans[2]),
    'B_CA1_1': fg.place_axes_on_grid(fig, xspan=xspans[3], yspan=yspans[0]),
    'B_CA1_2': fg.place_axes_on_grid(fig, xspan=xspans[3], yspan=yspans[1]),
    'B_CA1_3': fg.place_axes_on_grid(fig, xspan=xspans[3], yspan=yspans[2]),
    'B_CA1_1': fg.place_axes_on_grid(fig, xspan=xspans[4], yspan=yspans[0]),
    'B_CA1_2': fg.place_axes_on_grid(fig, xspan=xspans[4], yspan=yspans[1]),
    'B_CA1_3': fg.place_axes_on_grid(fig, xspan=xspans[4], yspan=yspans[2]),
    'B_LP_1': fg.place_axes_on_grid(fig, xspan=xspans[5], yspan=yspans[0]),
    'B_LP_2': fg.place_axes_on_grid(fig, xspan=xspans[5], yspan=yspans[1]),
    'B_LP_3': fg.place_axes_on_grid(fig, xspan=xspans[5], yspan=yspans[2]),
    'B_PO_1': fg.place_axes_on_grid(fig, xspan=xspans[6], yspan=yspans[0]),
    'B_PO_2': fg.place_axes_on_grid(fig, xspan=xspans[6], yspan=yspans[1]),
    'B_PO_3': fg.place_axes_on_grid(fig, xspan=xspans[6], yspan=yspans[2]),
}

labels = [{'label_text': 'a', 'xpos': get_label_pos(width, xspans[1][0]),
           'ypos': get_label_pos(height, yspans[0][0], pad=0.3),
           'fontsize': 10, 'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
          {'label_text': 'b', 'xpos': get_label_pos(width, xspans[2][0]),
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
axs['label2'].annotate('LDA colored by gt label', xy=(0, 0.5), xycoords='axes fraction', size='8', ha='right', va='center', rotation='vertical')
axs['label3'].set_axis_off()
axs['label3'].annotate('LDA colored by pred label', xy=(0, 0.5), xycoords='axes fraction', size='8', ha='right', va='center', rotation='vertical')

# Remove some axis
row2_xlims = []
row2_ylims = []
row3_xlims = []
row3_ylims = []
for key, ax in axs.items():
    if key[0] != 'B':
        continue
    if 'B_VIs' not in key:
        ax.set_ylabel('')
    if key[-1] == 2:
        ax.get_xlim()



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
# Change this to the institutes you use in the analysis
institutions = ['Berkeley', 'CCU', 'CSHL (C)', 'CSHL (Z)', 'NYU', 'Princeton', 'SWC', 'UCL',
                'UCLA', 'UW']
# Change this so the labs are roughly centered in the axis
offset = 0.05
plot_horizontal_institute_legend(institutions, axs['labs'], offset=offset)

fig.savefig('template.pdf')
# test to make things work
axs['A_1'].set_ylabel('Number of \npermutations')
axs['A_1'].set_xlabel('Macro F1')
axs['A_1'].set_title('Brain region (all neurons)')

axs['A_2'].set_ylabel('Dim2')
axs['A_2'].set_xlabel('Dim1')

fig.savefig('template.pdf')

# Figure 7 take 2
from reproducible_ephys_functions import (figure_style, get_row_coord, get_label_pos, plot_horizontal_institute_legend,
                                          BRAIN_REGIONS, REGION_RENAME)
import matplotlib.pyplot as plt
import figrid as fg


region_colors = figure_style(return_colors=True)
width = 7
height = 6
fig = plt.figure(figsize=(width, height))

xspans1 = get_row_coord(width, [5, 1, 5, 1])
xspans2 = get_row_coord(width, [1, 1, 1, 1, 1])
yspans = get_row_coord(height, [1, 1, 1, 1], hspace=0.6, pad=0.3)



axs = {
    'regions': fg.place_axes_on_grid(fig, xspan=xspans1[1], yspan=yspans[1]),
    'labs': fg.place_axes_on_grid(fig, xspan=xspans1[3], yspan=yspans[1]),
    'A_1': fg.place_axes_on_grid(fig, xspan=xspans1[0], yspan=yspans[0]),
    'A_2': fg.place_axes_on_grid(fig, xspan=xspans1[0], yspan=yspans[1]),
    'A_3': fg.place_axes_on_grid(fig, xspan=xspans1[2], yspan=yspans[0]),
    'A_4': fg.place_axes_on_grid(fig, xspan=xspans1[2], yspan=yspans[1]),
    'B_VIs_1': fg.place_axes_on_grid(fig, xspan=xspans2[0], yspan=yspans[2]),
    'B_VIs_2': fg.place_axes_on_grid(fig, xspan=xspans2[0], yspan=yspans[3]),
    'B_CA1_1': fg.place_axes_on_grid(fig, xspan=xspans2[1], yspan=yspans[2]),
    'B_CA1_2': fg.place_axes_on_grid(fig, xspan=xspans2[1], yspan=yspans[3]),
    'B_CA1_1': fg.place_axes_on_grid(fig, xspan=xspans2[2], yspan=yspans[2]),
    'B_CA1_2': fg.place_axes_on_grid(fig, xspan=xspans2[2], yspan=yspans[3]),
    'B_LP_1': fg.place_axes_on_grid(fig, xspan=xspans2[3], yspan=yspans[2]),
    'B_LP_2': fg.place_axes_on_grid(fig, xspan=xspans2[3], yspan=yspans[3]),
    'B_PO_1': fg.place_axes_on_grid(fig, xspan=xspans2[4], yspan=yspans[2]),
    'B_PO_2': fg.place_axes_on_grid(fig, xspan=xspans2[4], yspan=yspans[3]),
}

labels = [{'label_text': 'a', 'xpos': get_label_pos(width, xspans1[0][0]),
           'ypos': get_label_pos(height, yspans[0][0], pad=0.3),
           'fontsize': 10, 'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
          {'label_text': 'b', 'xpos': get_label_pos(width, xspans1[2][0]),
           'ypos': get_label_pos(height, yspans[0][0], pad=0.3), 'fontsize': 10,
           'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
          {'label_text': 'c', 'xpos': get_label_pos(width, xspans2[0][0]),
           'ypos': get_label_pos(height, yspans[2][0], pad=0.3), 'fontsize': 10,
           'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
          ]

fg.add_labels(fig, labels)

adjust = 0.3
fig.subplots_adjust(top=1 - adjust / height, bottom=(adjust - 0.2) / height, left=(adjust) / width,
                     right=1 - (adjust) / width)

axs['label1'].set_axis_off()
axs['label1'].annotate('Decoding', xy=(0, 0.5), xycoords='axes fraction', size='8', ha='right', va='center', rotation='vertical')
axs['label2'].set_axis_off()
axs['label2'].annotate('LDA colored by gt label', xy=(0, 0.5), xycoords='axes fraction', size='8', ha='right', va='center', rotation='vertical')
axs['label3'].set_axis_off()
axs['label3'].annotate('LDA colored by pred label', xy=(0, 0.5), xycoords='axes fraction', size='8', ha='right', va='center', rotation='vertical')

# Remove some axis
row2_xlims = []
row2_ylims = []
row3_xlims = []
row3_ylims = []
for key, ax in axs.items():
    if key[0] != 'B':
        continue
    if 'B_VIs' not in key:
        ax.set_ylabel('')
    if key[-1] == 2:
        ax.get_xlim()



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
# Change this to the institutes you use in the analysis
institutions = ['Berkeley', 'CCU', 'CSHL (C)', 'CSHL (Z)', 'NYU', 'Princeton', 'SWC', 'UCL',
                'UCLA', 'UW']
# Change this so the labs are roughly centered in the axis
offset = 0.05
plot_horizontal_institute_legend(institutions, axs['labs'], offset=offset)

fig.savefig('template.pdf')
# test to make things work
axs['A_1'].set_ylabel('Number of \npermutations')
axs['A_1'].set_xlabel('Macro F1')
axs['A_1'].set_title('Brain region (all neurons)')

axs['A_2'].set_ylabel('Dim2')
axs['A_2'].set_xlabel('Dim1')

fig.savefig('template.pdf')


### FIGURE (

from reproducible_ephys_functions import (figure_style, get_row_coord, get_label_pos, plot_vertical_institute_legend,
                                          BRAIN_REGIONS, REGION_RENAME)
import matplotlib.pyplot as plt
import figrid as fg
import numpy as np


region_colors = figure_style(return_colors=True)
width = 7
height = 7.3
fig = plt.figure(figsize=(width, height))

xspans = get_row_coord(width, [10, 1], hspace=0.2)
yspans = get_row_coord(height, [3, 2], hspace=0.8, pad=0.3)


axs = {
    'A': fg.place_axes_on_grid(fig, xspan=xspans[0], yspan=yspans[0], dim=[1, 5], wspace=0.3),
    'B': fg.place_axes_on_grid(fig, xspan=xspans[0], yspan=yspans[1], dim=[2, 5], wspace=0.3),
    'labs_a': fg.place_axes_on_grid(fig, xspan=xspans[1], yspan=yspans[0]),
    'regs_labs_b': fg.place_axes_on_grid(fig, xspan=xspans[1], yspan=yspans[1]),
}

labels = [{'label_text': 'a', 'xpos': get_label_pos(width, xspans[0][0]),
           'ypos': get_label_pos(height, yspans[0][0], pad=0.3),
           'fontsize': 10, 'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
          {'label_text': 'b', 'xpos': get_label_pos(width, xspans[0][0]),
           'ypos': get_label_pos(height, yspans[1][0], pad=0.3), 'fontsize': 10,
           'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
          ]

fg.add_labels(fig, labels)

adjust = 0.3
fig.subplots_adjust(top=1 - adjust / height, bottom=(adjust + 0.2) / height, left=(adjust + 0.2) / width,
                     right=1 - (adjust - 0.2) / width)

axs['labs_a'].set_axis_off()
institutions = ['Berkeley', 'CCU', 'CSHL (C)', 'CSHL (Z)', 'NYU', 'Princeton', 'SWC', 'UCL',
                'UCLA', 'UW']
plot_vertical_institute_legend(institutions, axs['labs_a'], span=(0.3, 0.7), offset=0)

axs['regs_labs_b'].set_axis_off()

# plot the regions
pos = np.linspace(0.7, 0.9, len(BRAIN_REGIONS))[::-1]
for p, reg in zip(pos, BRAIN_REGIONS):
    axs['regs_labs_b'].text(0, p, REGION_RENAME[reg], color=region_colors[reg], fontsize=7, transform=axs['regs_labs_b'].transAxes)

# plot the labs
institutions = ['Berkeley', 'CCU', 'CSHL (C)', 'CSHL (Z)', 'NYU', 'Princeton', 'SWC', 'UCL',
                'UCLA', 'UW']
plot_vertical_institute_legend(institutions, axs['regs_labs_b'], span=(0, 0.4), offset=0)




# Hyun figure 7
figure_style()
width = 7
height = 8.74
fig = plt.figure(figsize=(width, height))
xspans_row1 = get_row_coord(width, [1])
xspans_row2 = get_row_coord(width, [1, 1])
xspans_row3 = get_row_coord(width, [2, 3])
xspans_e = get_row_coord(width, [5, 1], hspace=0.1, pad=0, span=xspans_row3[1])
yspans = get_row_coord(height, [1.5, 1.5, 1], hspace=0.8, pad=0.3)

axs = {
    'A': fg.place_axes_on_grid(fig, xspan=xspans_row1[0], yspan=yspans[0]),
    'B': fg.place_axes_on_grid(fig, xspan=xspans_row2[0], yspan=yspans[1], dim=[3, 1], hspace=0.3),
    'C': fg.place_axes_on_grid(fig, xspan=xspans_row2[1], yspan=yspans[1], dim=[3, 2], hspace=0.4),
    'D': fg.place_axes_on_grid(fig, xspan=xspans_row3[0], yspan=yspans[2]),
    'E_1': fg.place_axes_on_grid(fig, xspan=xspans_e[0], yspan=yspans[2]),
    'E_2': fg.place_axes_on_grid(fig, xspan=xspans_e[1], yspan=yspans[2]),
}

labels = [{'label_text': 'a', 'xpos': get_label_pos(width, xspans_row1[0][0]),
           'ypos': get_label_pos(height, yspans[0][0], pad=0.3),
           'fontsize': 10, 'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
          {'label_text': 'b', 'xpos': get_label_pos(width, xspans_row2[0][0]),
           'ypos': get_label_pos(height, yspans[1][0], pad=0.3), 'fontsize': 10,
           'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
          {'label_text': 'c', 'xpos': get_label_pos(width, xspans_row2[1][0]),
           'ypos': get_label_pos(height, yspans[1][0], pad=0.3), 'fontsize': 10,
           'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
          {'label_text': 'd', 'xpos': get_label_pos(width, xspans_row3[0][0]),
           'ypos': get_label_pos(height, yspans[2][0], pad=0.3), 'fontsize': 10,
           'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
          {'label_text': 'e', 'xpos': get_label_pos(width, xspans_row3[1][0]),
           'ypos': get_label_pos(height, yspans[2][0], pad=0.3), 'fontsize': 10,
           'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
          ]

fg.add_labels(fig, labels)


axs['A'].set_axis_off()


axs['B'][0].set_ylim(0, 120)
axs['B'][0].set_ylabel('Firing rate (Hz)')
axs['B'][0].set_xlim(-0.5, 1)
axs['B'][0].set_xticks([-0.5, 0, 0.5, 1])
axs['B'][0].set_xticklabels([])
axs['B'][0].set_title('Left stimulus')
axs['B'][0].plot([-0.3, 0.7], [50, 50], 'k', label='Observed')
axs['B'][0].plot([-0.3, 0.7], [50, 50], 'r', label='Predicted')
axs['B'][0].legend(loc='upper left', fontsize=6, frameon=False)

axs['B'][1].annotate('Observed \n raster plot', xy=(-0.105, 0.5), xycoords='axes fraction',
                       size='8', ha='right', va='center', rotation='vertical')
axs['B'][1].set_ylabel('Trials')
axs['B'][1].set_yticks([])
axs['B'][1].set_yticklabels([])
axs['B'][1].set_xlim(-0.5, 1)
axs['B'][1].set_xticks([-0.5, 0, 0.5, 1])
axs['B'][1].set_xticklabels([])

axs['B'][2].annotate('Predicted \n raster plot', xy=(-0.105, 0.5), xycoords='axes fraction',
                     size='8', ha='right', va='center', rotation='vertical')
axs['B'][2].set_ylabel('Trials')
axs['B'][2].set_yticks([])
axs['B'][2].set_yticklabels([])
axs['B'][2].set_xlim(-0.5, 1)
axs['B'][2].set_xticks([-0.5, 0, 0.5, 1])
axs['B'][2].set_xticklabels([-0.5, 0, 0.5, 1])
axs['B'][2].set_xlabel('Time (s)')
# im = axs['B'][2].imshow(200 * np.random.random((20, 20)), extent=[0, 0.5, 20, 40], aspect='auto', cmap='binary')
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# divider = make_axes_locatable(axs['B'][2])
# colorbar_axes = divider.append_axes("right", size="100%", pad=0.1)
# cbar = fig.colorbar(im, cax=colorbar_axes, orientation='vertical')
#plt.colorbar(mappable=im, ax=axs['B'][2])

titles = ['Firing rate', 'Wheel velocity', 'Motion energy', 'Right paw speed', 'Nose tip speed', 'Licks']
for i, ax in enumerate(np.array(axs['C']).flatten()):

    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xlim(-0.5, 1)
    ax.set_xticks([-0.5, 0, 0.5, 1])
    ax.set_title(titles[i])
    if np.mod(i, 2) == 0:
        ax.set_ylabel('Trials')
    if i > 3:
        ax.set_xlabel('Time (s)')
    else:
        ax.set_xticklabels([])


dlabels = ['', 'Motion energy', 'Paw speed', 'Wheel velocity', 'Lick',
           'First movement', 'Stimuli', 'Go cue', 'Pupil diameter',
           'Nose speed', 'Reward', 'Choice', 'Mouse prior', 'Decision strategy',
           'Last mouse prior', 'Noise', '']
elabels1 = ['', 'Motion energy', 'Lick', 'Paw speed', 'Decision strategy',
            'Reward', 'Brain region', 'Waveform amplitude', 'First movement',
            'Nose speed', 'Mouse prior', 'Choice', 'Wheel velocity',
            'Last mouse prior', 'Stimuli', 'Session', 'Pupil diameter', 'z',
            'Go cue', 'x', 'y', 'Lab', 'Waveform width', 'Noise', '']
elabels2 = ['','Behavioral', 'Electrophysiological', 'Task-related', '']


axs['D'].set_xticks(np.arange(0, len(dlabels)))
axs['D'].set_xticklabels(dlabels, rotation=45, ha='right')
axs['D'].set_ylim(-0.2, 0.5)
axs['D'].set_ylabel('R$^2$')
axs['D'].set_title('Single-covariate analysis')

axs['E_1'].set_xticks(np.arange(0, len(elabels1)))
axs['E_1'].set_xticklabels(elabels1, rotation=45, ha='right')
axs['E_1'].set_ylim(-0.2, 0.5)
axs['E_1'].set_ylabel(r'$\Delta$R$^2$')
axs['E_1'].set_title('Leave-one-out analysis')


region_colors = {'LP': 'k', 'CA1': 'b', 'DG': 'r', 'PPC': 'g', 'PO': 'y'}

for i, reg in enumerate(BRAIN_REGIONS):
    if i == 0:
        text = axs['E_1'].text(0.5, 0.8, REGION_RENAME[reg], color=region_colors[reg], fontsize=8,
                       transform=axs['E_1'].transAxes)
    else:
        text = axs['E_1'].annotate(
            '  ' + REGION_RENAME[reg], xycoords=text, xy=(1, 0), verticalalignment="bottom",
            color=region_colors[reg], fontsize=8)


axs['E_2'].set_xticks(np.arange(0, len(elabels2)))
axs['E_2'].set_xticklabels(elabels2, rotation=45, ha='right')
axs['E_2'].set_ylim(-0.2, 0.5)
axs['E_2'].set_yticklabels([])
axs['E_2'].set_title('Leave-group-out')


adjust = 0.3
fig.subplots_adjust(top=1 - adjust / height, bottom=(adjust + 0.4) / height, left=(adjust + 0.2) / width,
                     right=1 - (adjust) / width)
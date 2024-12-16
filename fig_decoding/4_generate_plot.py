import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from scipy.cluster import hierarchy
from scipy.stats import f_oneway
from statsmodels.stats.multitest import multipletests
from reproducible_ephys_functions import (
    figure_style, 
    get_row_coord, 
    get_label_pos, 
    plot_vertical_institute_legend,
    BRAIN_REGIONS, 
    REGION_RENAME,
    LAB_MAP
)
import figrid as fg
from one.api import ONE


def load_dictionaries():
    """Load pre-saved dictionaries from .pkl files."""
    with open("region_f1_dict.pkl", "rb") as f:
        region_f1_dict = pickle.load(f)

    with open("region_pval_dict.pkl", "rb") as f:
        region_pval_dict = pickle.load(f)

    with open("lab_f1_dict.pkl", "rb") as f:
        lab_f1_dict = pickle.load(f)

    with open("lab_pval_dict.pkl", "rb") as f:
        lab_pval_dict = pickle.load(f)

    return region_f1_dict, region_pval_dict, lab_f1_dict, lab_pval_dict


def setup_figure_layout(width, height):
    """Set up the figure layout."""
    fig = plt.figure(figsize=(width, height))

    xspans = get_row_coord(width, [10, 1], hspace=0.2, pad=0.5)
    yspans = get_row_coord(height, [3, 2], hspace=0.8, pad=0.3)

    axs = {
        'A': fg.place_axes_on_grid(fig, xspan=xspans[0], yspan=yspans[0], dim=[1, 4], wspace=0.3),
        'B': fg.place_axes_on_grid(fig, xspan=xspans[0], yspan=yspans[1], dim=[2, 4], wspace=0.3),
        'labs_a': fg.place_axes_on_grid(fig, xspan=xspans[1], yspan=yspans[0]),
        'regs_labs_b': fg.place_axes_on_grid(fig, xspan=xspans[1], yspan=yspans[1]),
    }

    return fig, axs, xspans, yspans


def configure_labels(fig, xspans, yspans, width, height):
    """Add figure labels."""
    labels = [
        {'label_text': 'a', 'xpos': get_label_pos(width, xspans[0][0], pad=0.5),
         'ypos': get_label_pos(height, yspans[0][0], pad=0.3),
         'fontsize': 10, 'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
        {'label_text': 'b', 'xpos': get_label_pos(width, xspans[0][0], pad=0.5),
         'ypos': get_label_pos(height, yspans[1][0], pad=0.3), 'fontsize': 10,
         'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
    ]
    fg.add_labels(fig, labels)


# Panel A function with added grey dashed lines
def plot_panel_a(ax, varis, regs, tt, sb, nscores):
    ps = {}
    variable_titles = ["Choice", "Stimulus Side", "Reward", "Wheel Speed"]
    for k, vari in enumerate(varis):
        data_file = f'{vari}.parquet'
        d = pd.read_parquet(data_file)
        # Process data
        eids = [one.pid2eid(pid)[0] for pid in d['pid'].values]
        pths = one.eid2path(eids)
        d['lab'] = [b[str(p).split('/')[str(p).split('/').index('Subjects') - 1]] for p in pths]
        d['subject'] = [str(p).split('/')[str(p).split('/').index('Subjects') + 1] for p in pths]

        d = d.dropna(subset=['score', 'lab', 'region', 'subject'])

        if tt == 'stripplot':
            # Filter data
            filtered_data = d[d['region'].isin(regs)]
            
            sns.stripplot(
                x='score', y='region', hue=sb, data=filtered_data, jitter=True, dodge=True,
                ax=ax[k], order=regs, size=2
            )
            
            # Add grey dashed lines between regions
            for i in range(len(regs) - 1):
                ax[k].axhline(i + 0.5, color='grey', linestyle='--', linewidth=0.8)
            
            # Configure titles and labels
            ax[k].set_title(vari.capitalize())
            ax[k].set_xlabel("Score over chance level")

            # ANOVA analysis for stripplot
            labs = np.unique(d[sb].values)
            for reg in regs:
                scores_by_lab = [d[(d[sb] == lab) & (d['region'] == reg)]['score'].values for lab in labs]
                filtered_scores_by_lab = [lab_scores for lab_scores in scores_by_lab if lab_scores.size >= nscores]

                if len(filtered_scores_by_lab) < 2:
                    continue

                F, p = f_oneway(*filtered_scores_by_lab)
                ps[f"{vari}_{reg}"] = p
                m = np.max(np.concatenate(scores_by_lab))

                weight = 'bold' if p < 0.05 / len(labs) else 'normal'
                if vari == 'wheel-speed':
                    x = 0.35
                elif vari == 'stimside':
                    x = 0.35
                elif vari == 'whisker-motion-energy':
                    x = 0.5
                else:
                    x = 0.35
                
                # Add ANOVA results (F and p values) to the plot
                ax[k].set_xlim(-0.1, 0.55)
                ax[k].text(x, regs.index(reg), f'F={F:.2f}\np={p:.3f}', weight=weight, ha='left', va='center', fontsize=6)

                if k in [1, 2, 3]:
                    ax[k].set_yticklabels("")


        elif tt == 'mean_std':
            reg_stats = d.groupby('region')['score'].agg(
                mean_score=np.nanmean, std_score=np.nanstd, count_scores='count'
            ).reset_index()
            x = reg_stats['mean_score'].values
            y = reg_stats['std_score'].values
            ax[k].scatter(x, y, s=10, label=vari)
        ax[k].legend().set_visible(False)

        # Add variable title only for the first row
        if k < len(varis):
            ax[k].set_title(variable_titles[k], fontsize=7)

        if k == 0:
            ax[k].set_ylabel("Region")
        else:
            ax[k].set_ylabel("")


# Panel B function with updated variable titles
def plot_panel_b(ax, varis, region_f1_dict, lab_f1_dict, lab_pval_dict, region_cols):
    palette = [region_cols["CA1"], region_cols["DG"], region_cols["LP"], region_cols["PO"], region_cols["PPC"]]

    # Updated variable titles
    variable_titles = ["Choice", "Stimulus Side", "Reward", "Wheel Speed"]
    for i, vari in enumerate(varis):
        data_file = f'{vari}.parquet'
        d = pd.read_parquet(data_file)
        # Process data
        eids = [one.pid2eid(pid)[0] for pid in d['pid'].values]
        pths = one.eid2path(eids)
        d['lab'] = [b[str(p).split('/')[str(p).split('/').index('Subjects') - 1]] for p in pths]
        d['subject'] = [str(p).split('/')[str(p).split('/').index('Subjects') + 1] for p in pths]
        d = d.dropna(subset=['score', 'lab', 'region', 'subject'])

        ax = [item for sublist in axs['B'] for item in sublist]

        # Pass the correct Axes instance
        sns.scatterplot(
            data=d, x="unitcount", y="score", hue="region", palette=palette, ax=ax[i], marker='x', legend=False,
            linewidth=.35, size=.3,
        )
        sns.scatterplot(
            data=d, x="unitcount", y="score", hue="lab", ax=ax[i + len(varis)], marker='x', legend=False,
            linewidth=.35, size=.3,
        )

        # Remove legends explicitly (in case they still appear)
        if ax[i].legend_:
            ax[i].legend_.remove()
        if ax[i + len(varis)].legend_:
            ax[i + len(varis)].legend_.remove()


        # Add variable title only for the first row
        if i < len(varis):
            ax[i].set_title(variable_titles[i], fontsize=7)

        # Move F1 and p-values to the top-right corner inside the subplot
        if region_pval_dict[vari] < 0.01:
            ax[i].text(
                0.95, 0.95, f"F1={region_f1_dict[vari]:.2f}\np<0.01",
                fontsize=6, ha="right", va="top", transform=ax[i].transAxes, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
            )
        else:
            ax[i].text(
                0.95, 0.95, f"F1={region_f1_dict[vari]:.2f}\np={region_pval_dict[vari]:.2f}",
                fontsize=6, ha="right", va="top", transform=ax[i].transAxes, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
            )
        ax[-1].set_xlabel("Number Of Neurons", fontsize=7)
        ax[i + len(varis)].text(
            0.95, 0.95, f"F1={lab_f1_dict[vari]:.2f}\np={lab_pval_dict[vari]:.2f}",
            fontsize=6, ha="right", va="top", transform=ax[i + len(varis)].transAxes, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )
        ax[i + len(varis)].set_xlabel("Number of neurons", fontsize=7)

        # Set ylabel only for subplots in the first column (e.g., 0, 4)
        if i % 4 == 0:  # Adjust the index to your grid structure
            ax[i].set_ylabel("Score over chance level")
            ax[i + len(varis)].set_ylabel("Score over chance level")
        else:
            ax[i].set_ylabel("")
            ax[i + len(varis)].set_ylabel("")

        if i % 4 == 0:
            ax[i].set_ylim(-0.075, 0.4)
            ax[i + len(varis)].set_ylim(-0.075, 0.4)
        elif i % 4 == 1:
            ax[i].set_ylim(-0.075, 0.4)
            ax[i + len(varis)].set_ylim(-0.075, 0.4)
        elif i % 4 == 2:
            ax[i].set_ylim(-0.075, 0.4)
            ax[i + len(varis)].set_ylim(-0.075, 0.4)
        else:
            ax[i].set_ylim(-0.05, 0.4)
            ax[i + len(varis)].set_ylim(-0.05, 0.4)

        ax[i].set_xlim(-5, 105)
        ax[i + len(varis)].set_xlim(-5, 105)


if __name__ == "__main__":

    ONE.setup(
        base_url="https://openalyx.internationalbrainlab.org", silent=True,
    )
    one = ONE(password='international')

    varis = ['choice', 'stimside', 'reward', 'wheel-speed']
    region_colors = figure_style(return_colors=True)
    _, b, lab_cols = LAB_MAP()

    # Load data
    region_f1_dict, region_pval_dict, lab_f1_dict, lab_pval_dict = load_dictionaries()

    # Figure setup
    width = 7
    height = 7.3
    fig = plt.figure(figsize=(width, height))

    xspans = get_row_coord(width, [10, 1], hspace=0.2, pad=0.5)
    yspans = get_row_coord(height, [3, 2], hspace=0.8, pad=0.3)

    axs = {
        'A': fg.place_axes_on_grid(fig, xspan=xspans[0], yspan=yspans[0], dim=[1, 4], wspace=0.3),
        'B': fg.place_axes_on_grid(fig, xspan=xspans[0], yspan=yspans[1], dim=[2, 4], wspace=0.3),
        'labs_a': fg.place_axes_on_grid(fig, xspan=xspans[1], yspan=yspans[0]),
        'regs_labs_b': fg.place_axes_on_grid(fig, xspan=xspans[1], yspan=yspans[1]),
    }

    # Add figure labels
    labels = [{'label_text': 'a', 'xpos': get_label_pos(width, xspans[0][0], pad=0.5),
           'ypos': get_label_pos(height, yspans[0][0], pad=0.3),
           'fontsize': 10, 'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
          {'label_text': 'b', 'xpos': get_label_pos(width, xspans[0][0], pad=0.5),
           'ypos': get_label_pos(height, yspans[1][0], pad=0.3), 'fontsize': 10,
           'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
          ]
    fg.add_labels(fig, labels)

    # Plot institutions and regions
    axs['labs_a'].set_axis_off()
    institutions = ['Berkeley', 'CCU', 'CSHL (C)', 'CSHL (Z)', 'NYU', 'Princeton', 'SWC', 'UCL',
                    'UCLA', 'UW']
    plot_vertical_institute_legend(institutions, axs['labs_a'], span=(0.3, 0.7), offset=0)

    axs['regs_labs_b'].set_axis_off()
    pos = np.linspace(0.7, 0.9, len(BRAIN_REGIONS))[::-1]
    for p, reg in zip(pos, BRAIN_REGIONS):
        axs['regs_labs_b'].text(0, p, REGION_RENAME[reg], color=region_colors[reg], fontsize=7, transform=axs['regs_labs_b'].transAxes)
    plot_vertical_institute_legend(institutions, axs['regs_labs_b'], span=(0, 0.4), offset=0)

    # Adjust figure layout
    adjust = 0.3
    fig.subplots_adjust(top=1 - adjust / height, bottom=(adjust + 0.2) / height, left=(adjust) / width,
                     right=1 - (adjust - 0.2) / width)

    plot_panel_a(ax=axs['A'], varis=varis, regs=['VISa', 'CA1', 'DG', 'LP', 'PO'], tt='stripplot', sb='lab', nscores=3)
    plot_panel_b(ax=axs['B'], varis=varis, region_f1_dict=region_f1_dict, lab_f1_dict=lab_f1_dict, lab_pval_dict=lab_pval_dict, region_cols=region_colors)

    # Save and show the plot
    plt.savefig("decoding_analyses.png", dpi=500)

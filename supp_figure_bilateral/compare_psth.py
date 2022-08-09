import matplotlib.pyplot as plt
import numpy as np
from supp_figure_bilateral.load_data import load_neural_data, load_dataframe
from reproducible_ephys_functions import filter_recordings, BRAIN_REGIONS, labs


lab_number_map, institution_map, lab_colors = labs()
probe_colors = {'probe00': 'blue', 'probe01': 'red'}

def plot_comparative_psth():
    data = load_neural_data(event='move', norm='subtract', smoothing='sliding')
    df_chns = load_dataframe(df_name='neural')
    df_filt = filter_recordings(df_chns, bilateral=True)
    all_frs = data['all_frs'][df_filt['include'] == 1]
    all_frs_std = data['all_frs_std'][df_filt['include'] == 1]
    df_filt = df_filt[df_filt['include'] == 1].reset_index()
    df_filt_reg = df_filt.groupby('region')

    fig, ax = plt.subplots(1, len(BRAIN_REGIONS), figsize=(16 * 0.99, 9 * 0.99))

    for iR, reg in enumerate(BRAIN_REGIONS):
        df_reg = df_filt_reg.get_group(reg)
        df_reg = df_reg.groupby('subject')
        for subj in df_reg.groups.keys():
            color = next(plt.gca()._get_lines.prop_cycler)['color']
            df_reg_subj = df_reg.get_group(subj)
            df_reg_subj_probe = df_reg_subj.groupby('probe')
            for probe in df_reg_subj_probe.groups.keys():
                probe_idx = df_reg_subj_probe.groups[probe]
                frs_probe = all_frs[probe_idx, :]
                ax[iR].plot(data['time'], np.mean(frs_probe, axis=0), c=color,
                            alpha=0.8)
        ax[iR].set_ylim(bottom=-3, top=11)
        ax[iR].axvline(0, color='k', ls='--')
        ax[iR].spines["right"].set_visible(False)
        ax[iR].spines["top"].set_visible(False)
        ax[iR].set_xlim(left=data['time'][0], right=data['time'][-1])
        if iR >= 1:
            ax[iR].set_yticklabels([])
        else:
            ax[iR].set_ylabel("Baselined firing rate (sp/s)")
            #if len(plotted_regions) != 1:
                #ax[iR].set_ylabel("Baselined firing rate (sp/s)")
                #ax[iR].set_xlabel("Time (s)")
        ax[iR].set_title(reg)
        if iR == 1 or len(BRAIN_REGIONS) == 1:
            ax[iR].set_xlabel("Time from movement onset (s)")
    plt.tight_layout()
    plt.savefig("temp bilateral figure compact right")
    plt.show()

    return df_filt

if __name__ == '__main__':
    df = plot_comparative_psth()

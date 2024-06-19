clear,close all, clc

data_path = '/Users/mt/Downloads/FlatIron/paper_repro_ephys_data/figure_spatial/';
save_path = '/Users/mt/Downloads/FlatIron/paper_repro_ephys_data/figure_spatial/figures/';

CSVfile = [data_path, 'fig_spatial_dataframe_filt.csv']; 

BrainRegions = ["PPC", "CA1", "DG", "LP", "PO"];

for br = BrainRegions
    fig_spatial_generate(CSVfile, br, save_path)
    close all
    %Can use function 'fig_spatial_generate_allTMtests' to examine all TM tests
end

fig_spatial_FanoFactor_generate(CSVfile, BrainRegions, save_path)
close all
fig_spatial_spike_waveforms(CSVfile, save_path)
close all

clear,close all, clc
data_path = '/Users/mt/Downloads/FlatIron/paper_repro_ephys_data/figure_spatial/';
save_path = '/Users/mt/Downloads/FlatIron/paper_repro_ephys_data/figure_spatial/figures/';


CSVfile = [data_path, 'fig_spatial_dataframe_filt.csv'];

BrainRegions = ["PPC", "CA1", "DG", "LP", "PO"];

fig_spatial_FanoFactor_generate(CSVfile, BrainRegions, save_path)


%%Plots over time:
%FanoFactorVsT_generate(CSVfile, BrainRegions, save_path)


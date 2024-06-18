clear,close all, clc
data_path = '/Users/mt/Downloads/FlatIron/paper_repro_ephys_data/figure_spatial/';
%%'C:\Users\Mayo\Downloads\ONE\openalyx.internationalbrainlab.org\paper_repro_ephys_data\figure7\'

save_path = '/Users/mt/Downloads/FlatIron/paper_repro_ephys_data/figure_spatial/figures/';
%%'C:\Users\Mayo\Downloads\ONE\openalyx.internationalbrainlab.org\paper_repro_ephys_data\figure7\figures\'

CSVfile = [data_path, 'figure_spatial_dataframe_filt.csv']; 

%BrainRegions = ["PPC", "CA1", "DG", "LP", "PO"];
BrainRegions = ["DG"]

for br = BrainRegions
Fig3Dplots_generate(CSVfile, br, save_path)
% close all
end

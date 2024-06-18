clear,close all, clc
data_path = '/Users/mt/Downloads/FlatIron/paper_repro_ephys_data/figure_spatial/';
%%'C:\Users\Mayo\Downloads\ONE\openalyx.internationalbrainlab.org\paper_repro_ephys_data\figure_spatial\'

save_path = '/Users/mt/Downloads/FlatIron/paper_repro_ephys_data/figure_spatial/figures/';
%%'C:\Users\Mayo\Downloads\ONE\openalyx.internationalbrainlab.org\paper_repro_ephys_data\figure_spatial\figures\'

CSVfile = [data_path, 'figure_spatial_dataframe_filt.csv']; 

%BrainRegions = ["PPC", "CA1", "DG", "LP", "PO"];
BrainRegions = ["DG"]

for br = BrainRegions
    fig_spatial_generate(CSVfile, br, save_path)
    %Can use function 'fig_spatial_generate_allTMtests' to examine all TM tests
end

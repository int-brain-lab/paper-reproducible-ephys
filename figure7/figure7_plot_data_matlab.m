clear,close all, clc
%data_path = '/Users/mt/Downloads/FlatIron/paper_repro_ephys_data/fig5and7CSVfiles_May82022/';
%data_path = '/Users/mt/Downloads/FlatIron/paper_repro_ephys_data/figure5/';
data_path = '/Users/mt/Downloads/FlatIron/paper_repro_ephys_data/fig_taskmodulation/';
%data_path = '/Users/mt/Downloads/FlatIron/paper_repro_ephys_data/figure7/';
%%'C:\Users\Mayo\Downloads\ONE\openalyx.internationalbrainlab.org\paper_repro_ephys_data\figure7\'

%save_path = '/Users/mt/Downloads/FlatIron/paper_repro_ephys_data/figure7/figuresMatlab/';
%save_path = '/Users/mt/Downloads/FlatIron/paper_repro_ephys_data/figure7/figuresMatlab_temp/';
save_path = '/Users/mt/Downloads/FlatIron/paper_repro_ephys_data/figure7/figures_2023/';
%%'C:\Users\Mayo\Downloads\ONE\openalyx.internationalbrainlab.org\paper_repro_ephys_data\figure7\figures\'

CSVfile = [data_path, 'fig_taskmodulation_dataframe.csv'];
%CSVfile = [data_path, 'figure7_dataframe.csv']; %'figure5_dataframe.csv'];

% BrainRegions = ["PPC", "CA1", "DG", "LP", "PO"];
BrainRegions = ["PPC"]

for br = BrainRegions
Fig3Dplots_generate(CSVfile, br, save_path)
% close all
end

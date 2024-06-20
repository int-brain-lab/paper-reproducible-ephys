clear,close all, clc
data_path = '/Users/mt/Downloads/FlatIron/paper_repro_ephys_data/fig_taskmodulation/'; %figure7/';
%'C:\Users\Mayo\Downloads\ONE\openalyx.internationalbrainlab.org\paper_repro_ephys_data\figure7\'
save_path = '/Users/mt/Downloads/FlatIron/paper_repro_ephys_data/figure7/figures_2023/'; %figuresMatlab/';
%'C:\Users\Mayo\Downloads\ONE\openalyx.internationalbrainlab.org\paper_repro_ephys_data\figure7\figures\'
CSVfile = [data_path, 'fig_taskmodulation_dataframe.csv'];%'figure7_dataframe.csv']; %'figure7_dataframe.csv'];

%BrainRegions = ["PPC", "CA1", "DG", "LP", "PO"];
BrainRegions = ["PPC"]; %["LP"];

%for br = BrainRegions
Fig3D_FanoFactor_plots_generate(CSVfile, BrainRegions, save_path)
% close all
%end

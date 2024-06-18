clear,close all, clc
data_path = '/Users/mt/Downloads/FlatIron/paper_repro_ephys_data/figure5/';
%'C:\Users\Mayo\Downloads\ONE\openalyx.internationalbrainlab.org\paper_repro_ephys_data\figure7\'
save_path = '/Users/mt/Downloads/FlatIron/paper_repro_ephys_data/figure7/figuresMatlab/TMtests_all';
%'C:\Users\Mayo\Downloads\ONE\openalyx.internationalbrainlab.org\paper_repro_ephys_data\figure7\figures\'
CSVfile = [data_path, 'figure5_dataframe.csv']; %'figure7_dataframe.csv'];

% BrainRegions = ["PPC", "CA1", "DG", "LP", "PO"];
BrainRegions = ["CA1", "DG"]; %["LP"]

for br = BrainRegions
Fig3Dplots_generate_allTMtests(CSVfile, br, save_path)
% close all
end
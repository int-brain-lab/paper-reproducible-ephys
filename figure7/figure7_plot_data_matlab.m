data_path = 'C:\Users\Mayo\Downloads\ONE\openalyx.internationalbrainlab.org\paper_repro_ephys_data\figure7\'
save_path = 'C:\Users\Mayo\Downloads\ONE\openalyx.internationalbrainlab.org\paper_repro_ephys_data\figure7\figures\'
CSVfile = [data_path, 'figure7_dataframe.csv'];

% BrainRegions = ["PPC", "CA1", "DG", "LP", "PO"];
BrainRegions = ["LP"]

for br = BrainRegions
Fig3Dplots_generate(CSVfile, br, save_path)
% close all
end
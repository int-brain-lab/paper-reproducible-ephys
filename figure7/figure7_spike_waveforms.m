clear, close all, clc
data_path = '/Users/mt/Downloads/FlatIron/paper_repro_ephys_data/fig_taskmodulation/';
%data_path = '/Users/mt/Downloads/FlatIron/paper_repro_ephys_data/figure7/';

save_path = '/Users/mt/Downloads/FlatIron/paper_repro_ephys_data/figure7/figures_2023/';

CSVfile = [data_path, 'fig_taskmodulation_dataframe.csv'];
%CSVfile = [data_path, 'figure7_dataframe.csv']; %'figure5_dataframe.csv'];

BrainRegions = ["PPC", "CA1", "DG", "LP", "PO"];
%BrainRegions = ["PPC", "LP", "PO"];
ThreshRegion = [0.35, 0.35, 0.35, 0.3, 0.3];

countBR = 0;
T = readtable(CSVfile);
for br = BrainRegions
    countBR = countBR+1;
    
    %Find neurons in specified brain region & their FR:
    Neur_idx = find(strcmp(T.region, br));
    AvgFR = T.avg_fr(Neur_idx);
    
    % Spike amp and duration:
    amps = T.amp(Neur_idx);
    width_p2t = T.p2t(Neur_idx);
    
    %NeuronInfo = [AvgFR, amps, width_p2t];
    
    thresh = ThreshRegion(countBR);
    figure(countBR)
    subplot(2,1,1)
    h1=histogram(width_p2t, 'binwidth', 0.04, 'normalization', 'probability', 'linewidth', 2);
    hold on
    Yl = get(gca, 'ylim');
    rectangle('Position', [thresh+0.005, 0, (h1.BinEdges(end)-thresh), Yl(2)],...
        'linewidth', 2, 'EdgeColor', [0, 0.7, 0.7, 1], 'FaceColor', [0, 0.7, 0.7, 0.1])%'FaceAlpha' is determined by the 4th #
    rectangle('Position', [0, 0, thresh, Yl(2)],...
        'linewidth', 2, 'EdgeColor', [0, 0, 0, 1], 'FaceColor', [0.5, 0.5, 0.5, 0.1])%'FaceAlpha' is determined by the 4th #
    %line([thresh, thresh], get(gca, 'ylim'), 'linewidth', 2, 'color', [0, 0.7, 0.7])
    set(gca, 'fontsize', 15)
    xlabel('Spike Width (ms)')
    ylabel('Probability Density')
    title(br)
    
    
    % 3 categories:
    width1 = [min(width_p2t),0];
    AvgFR1 = AvgFR(width_p2t>=width1(1) & width_p2t<=width1(2));
    width2 = [0, thresh];
    AvgFR2 = AvgFR(width_p2t>width2(1) & width_p2t<=width2(2));
    width3 = [thresh, max(width_p2t)];
    AvgFR3 = AvgFR(width_p2t>width3(1) & width_p2t<=width3(2));
    
    figure(countBR)
    subplot(2,1,2)
    [f1,x1] = ecdf(AvgFR1);
    [f2,x2] = ecdf(AvgFR2);
    [f3,x3] = ecdf(AvgFR3);
    plot(x2,f2, 'color', 'k', 'linewidth', 2)
    hold on
    plot(x3,f3, 'color', [0, 0.7, 0.7], 'linewidth', 2)
    plot(x1,f1, 'color', [0.5, 0.5, 0.5], 'linewidth', 1, 'linestyle', '-.')
    legend('Narrow Spike Width','Wide Spike Width', 'Negative Spike Width')
    set(gca, 'fontsize', 15)
    xlabel('Avg F.R. (sp/s)')
    ylabel('Cumulative Probability')
    
    
    fig = figure(countBR);
    saveas(fig, append(save_path, br, '_RSvsFS.png'));
    
    % figure(3); subplot(1,2,1)
    % ecdf(width_p2t);
    % subplot(1,2,2)
    % ecdf(amps);
    
    
    %General histograms for all brain regions:
    figure(10)
    subplot(2,3,countBR)
    h=histogram(width_p2t, 'binwidth', 0.04, 'normalization', 'probability', 'linewidth', 2);
    set(gca, 'fontsize', 14)
    xlabel('Spike Width (ms)')
    if countBR==1 || countBR==4
        ylabel('Probability Density')
    end
    title(br)

    figure(11)
    subplot(2,3,countBR)
    h=histogram(amps, 'binwidth', 2e-5, 'normalization', 'probability', 'linewidth', 2);
    set(gca, 'fontsize', 14)
    xlabel('Spike Amp')
    if countBR==1 || countBR==4
        ylabel('Probability Density')
    end
    title(br)
    
    
    figure(12)
    subplot(2,3,countBR)
    plot(width_p2t, amps, 'o', 'markersize', 3)
    xlabel('Spike Width')
    if countBR==1 || countBR==4
        ylabel('Spike Amp')
    end
    title(br)
    
    
    figure(13)
    subplot(2,3,countBR)
    h=histogram(log10(AvgFR), 'binwidth', 0.1, 'normalization', 'probability', 'linewidth', 2);
    set(gca, 'fontsize', 14)
    xlabel('Avg FR (sp/s)')
    if countBR==1 || countBR==4
        ylabel('Probability Density')
    end
    title(br)    
    xticklabels(num2cell(10.^get(gca,'XTick')));

    
    figure(14)
    subplot(2,3,countBR)
    h=histogram(AvgFR, 'binwidth', 2, 'linewidth', 2);%'normalization', 'probability', 
    set(gca, 'fontsize', 14)
    xlabel('Avg FR (sp/s)')
    if countBR==1 || countBR==4
        ylabel('count')%'Probability Density')
    end
    title(br)    
    
end


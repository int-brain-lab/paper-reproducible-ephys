% Code for generating average FR 3D plots and examining spatial position 
% and spike waveform characteristics as a source of variability (e.g., in 
% Fig. 7b).
% Code written by: Marsa Taheri

function [hCB, jitt, FRthresh, pMWU_corrected,...
    mdl, mdl_shuffle, FRthreshLow] = plot3D_FR(CSVfile, BrainRegion, ax, save_path)

%mdl and mdl_shuffle are the linear regression model for real and shuffled
%data
%jitt is the jitter added to xyz position; to keep it consistent between 3D
%plots of the same brain region, this value is an output.

T = readtable(CSVfile);

%AvgFR_col = find(strcmp(T.Properties.VariableNames, 'avg_fr')); 
X_col = find(strcmp(T.Properties.VariableNames, 'x')); 
Y_col = find(strcmp(T.Properties.VariableNames, 'y')); 
Z_col = find(strcmp(T.Properties.VariableNames, 'z')); 

%Find neurons in specified brain region:
Neur_idx = find(strcmp(T.region, BrainRegion));

%% Calculate distance from planned target center of mass and add jitter:

% Center of mass of target for RS brain regions, calculated previously:
CenterOfMassAll = [-0.002133610000000  -0.002000000000000  -0.000526236000000
  -0.001898770000000  -0.002000000000000  -0.001402680000000
  -0.001728650000000  -0.002000000000000  -0.002037580000000
  -0.001557180000000  -0.002000000000000  -0.002677510000000
  -0.001375230000000  -0.002000000000000  -0.003356550000000];

T_target = table(CenterOfMassAll(:,1), CenterOfMassAll(:,2), CenterOfMassAll(:,3),...
    'VariableNames',{'X0','Y0', 'Z0'},'RowNames',{'PPC','CA1', 'DG', 'LP','PO'});

X0_region = T_target.X0(find(strcmp(T_target.Row, BrainRegion)));
Y0_region = T_target.Y0(find(strcmp(T_target.Row, BrainRegion)));
Z0_region = T_target.Z0(find(strcmp(T_target.Row, BrainRegion)));


%Delta X, Y, Z position of each neuron within the desired brain region:
dXYZ = [T.x(Neur_idx) - X0_region, T.y(Neur_idx) - Y0_region, T.z(Neur_idx) - Z0_region]*1e6; %convert to microns

% Add x, y, z jitter:
jittSize = ones(1,3)*min([range(abs(dXYZ(:,1))), range(abs(dXYZ(:,2))), range(abs(dXYZ(:,3)))])./70;
jitt = jittSize.*(rand(size(dXYZ(:,1), 1), 3)-0.5); %center the noise around 0 (+ and -) with - 0.5

dXjitt = dXYZ(:,1) + jitt(:,1);
dYjitt = dXYZ(:,2) + jitt(:,2);
dZjitt = dXYZ(:,3) + jitt(:,3);


%% Make 3D plot
SmallDotSz = 13; LargeDotSz = SmallDotSz*3.8; FontSz=12;

AvgFR = T.avg_fr(Neur_idx);

% Identify regular and outlier neurons:
FiltThresh = 0.15;
DeviationFromMedian = (AvgFR - median(AvgFR))/range(AvgFR);

UnitsRegular = DeviationFromMedian<FiltThresh & DeviationFromMedian > -FiltThresh;
UnitsOutliers = DeviationFromMedian>=FiltThresh | DeviationFromMedian<= -FiltThresh;
% Separate outliers into HF and LF:
%UnitsHF = DeviationFromMedian>=FiltThresh; %High firing (HF) units
UnitsLF = DeviationFromMedian<= -FiltThresh; %Low firing (LF) units


FRthresh = range(AvgFR)*FiltThresh +  median(AvgFR);
if sum(UnitsLF)>0
    FRthreshLow = range(AvgFR)*(-FiltThresh) +  median(AvgFR); %thresh for low firing neurons
else
    FRthreshLow=NaN;
end

%--------To examine FR histogram and thresholds for outliers-------------
% figure
% hFR = histogram(AvgFR, 'binwidth', 0.12)%, 'normalization', 'probability');
% m1=mean(AvgFR); m2=median(AvgFR);
% p90=prctile(AvgFR, 90); p85=prctile(AvgFR, 85);
% p10=prctile(AvgFR, 10); p15=prctile(AvgFR, 15);
% FRthresh = range(AvgFR)*FiltThresh +  median(AvgFR);
% FRthreshLow = range(AvgFR)*(-FiltThresh) +  median(AvgFR);
% line([m1 m1], [min(hFR.Values), max(hFR.Values)], 'color', 'k')
% text(m1, max(hFR.Values), 'mean', 'color', 'k')
% line([m2 m2], [min(hFR.Values), max(hFR.Values)], 'color', 'r')
% text(m2, 0.85*max(hFR.Values), 'median', 'color', 'r')
% line([p90 p90], [min(hFR.Values), 1.1*max(hFR.Values)], 'color', 'b')
% text(p90, 1.1*max(hFR.Values), '90th perc', 'color', 'b')
% line([p10 p10], [min(hFR.Values), 1.1*max(hFR.Values)], 'color', 'b')
% text(p10, 1.1*max(hFR.Values), '10th perc', 'color', 'b')
% %line([p85 p85], [min(hFR.Values), max(hFR.Values)], 'color', [0.5 0.5 0.5])
% %text(p85, max(hFR.Values), '85th perc', 'color', [0.5 0.5 0.5])
% %line([p15 p15], [min(hFR.Values), max(hFR.Values)], 'color', [0.5 0.5 0.5])
% %text(p15, max(hFR.Values), '15th perc', 'color', [0.5 0.5 0.5])
% line([FRthresh FRthresh], [min(hFR.Values), max(hFR.Values)], 'color', [0.6 0 0.6],...
%     'linewidth', 2)
% text(FRthresh, 0.8*max(hFR.Values), 'FR Cut-off', 'color', [0.6 0 0.6])
% line([FRthreshLow FRthreshLow], [min(hFR.Values), max(hFR.Values)], 'color', [0.6 0 0.6],...
%     'linewidth', 2)
% %text(manualCutOff2, 0.8*max(hFR.Values), 'Manual Cut-off', 'color', [0.6 0 0.6])
% ylabel('count'); xlabel('FR (sp/sec)')% Deviation (normalized to -1 to 1)')
% set(gca,'fontsize', 14)
% title(BrainRegion)
%---------------------------------------------------------------------

AvgFR_regular = log10(AvgFR(UnitsRegular)');
dX_regular = dXjitt(UnitsRegular);
dY_regular = dYjitt(UnitsRegular);
dZ_regular = dZjitt(UnitsRegular);
scatter3(dX_regular, dY_regular, dZ_regular, SmallDotSz, AvgFR_regular,...
    'filled', 'MarkerFaceAlpha', 0.6,...
    'markeredgecolor', [1 1 1], 'markeredgealpha', 0.9, 'linewidth', 0.8)

hold on %plotting the target center of mass
sc=scatter3(0,0,0, 'x', 'linewidth', 2, 'markeredgecolor', [0.8 0.2 0.2]);
sc.SizeData=70;
sc=scatter3(0,0,0, 'or', 'markerfacecolor', 'r', 'markerfacealpha', 0.25);
sc.SizeData=55;
axis tight

AvgFR_outlier =log10(AvgFR(UnitsOutliers)');
dX_outlier = dXjitt(UnitsOutliers);
dY_outlier = dYjitt(UnitsOutliers);
dZ_outlier = dZjitt(UnitsOutliers);

scatter3(dX_outlier, dY_outlier, dZ_outlier, LargeDotSz, AvgFR_outlier','filled',...
    'markerfacealpha', 0.7,'markeredgecolor','k', 'linewidth', 0.8);
% Set x, y, z ticks:
XtickAuto = get(gca, 'Xtick'); YtickAuto = get(gca, 'Ytick'); ZtickAuto = get(gca, 'Ztick'); 
set(gca, 'Xtick', min(XtickAuto):200:max(XtickAuto),...
    'Ytick', min(YtickAuto):200:max(YtickAuto), 'Ztick', min(ZtickAuto):200:max(ZtickAuto))

colormap(ax, parula)
caxis manual
hCB=colorbar;
hCB.Title.String = {'      Avg FR'; '      (spikes/sec)'};
hCB.Title.FontSize = 12; %16;
xlabel('\DeltaX (\mum)'); ylabel('\DeltaY (\mum)'); zlabel('\DeltaZ (\mum)')
set(gca, 'fontsize', FontSz)

FRticks = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]; %to log10
caxis([log10(min(AvgFR)), log10(max(AvgFR))])
% To include tick for threshold, but no label:
hCB.Ticks = log10(sort([FRticks, FRthresh]));
% To exclude the label for the threshold value: 
TickLabelsTemp = 10.^(hCB.Ticks);
TickLabelsCell = num2cell(TickLabelsTemp);
FRthresh_idx = find(abs(10.^(hCB.Ticks) - FRthresh) == min(abs(10.^(hCB.Ticks) - FRthresh)));
TickLabelsCell{FRthresh_idx} = [];
set(hCB,'TickLabels', TickLabelsCell)
% To include the label for the threshold value:
%hCB.TickLabels = 10.^(hCB.Ticks);
% To have no threshold indicated:
hCB.Ticks = log10(FRticks); hCB.TickLabels = 10.^(hCB.Ticks);
%If low firing neurons are also outliers, place a line there:
if sum(UnitsLF)>0
    FRthreshLow = range(AvgFR)*(-FiltThresh) +  median(AvgFR); %thresh for low firing neurons
end

% % Annotate the colorbar to separate regular and outlier neurons:
% % Find position of colobar and its min/max values:
% barPos = get(hCB, 'Position');
% cLimits = caxis();
% % Find vertical position of FRthresh on colorbar:
% PositionRatio = (log10(FRthresh)-cLimits(1))/diff(cLimits);
% YlineLoc = barPos(2) + barPos(4)*PositionRatio;
% % Select horizontal position of line, centered on colorbar, and plot line:
% XlineLoc = barPos(1) + barPos(3)/2 + [0.022, -0.022];
% h_colorbarLine = annotation('line', XlineLoc, [YlineLoc, YlineLoc], 'Color', [0.9 0.6 0.2], 'linewidth', 2.5);                

% Sets the azimuth and elevation angles of the view:
if strcmp(BrainRegion, 'PPC')
    view(-25,13)
elseif strcmp(BrainRegion, 'CA1')
    view(-19, 13)
elseif strcmp(BrainRegion, 'DG')
    view(-18,12)
elseif strcmp(BrainRegion, 'LP')
    view(-17,24)
elseif strcmp(BrainRegion, 'PO')
    view(-24,18)
end
%To get azimuth and elevation of the current view: [caz,cel] = view()
hold off

%% Separate the two groups, then statistics (what's unique about outlier clusters):

% x, y, z histograms:
fig = figure;%(2)
for spHist=1:3
    subplot(3,1,spHist)
    Sample1(:,spHist) = dXYZ(UnitsRegular, spHist);
    Sample2(:,spHist) = dXYZ(UnitsOutliers, spHist);
    
    h1=histogram(Sample1(:,spHist),...
        'binwidth',100, 'edgecolor', [0.02, 0.31, 0.51],...
        'normalization', 'probability', 'DisplayStyle', 'stairs', 'linewidth',2, 'edgealpha', 0.8); hold on;
    hold on
    h2=histogram(Sample2(:,spHist),...
        'normalization', 'probability',...
        'binwidth',100, 'edgecolor', [1, 0.53, 0],...
        'facealpha', 0.4, 'edgealpha', 0.6, 'DisplayStyle', 'stairs', 'linewidth',3);

    if spHist==1
        xlabel('\DeltaX (L-M)')
        title([char(BrainRegion), ': ', num2str(sum(UnitsOutliers)),...
            ' Outlier neurons from ',...
            num2str(size(dXYZ,1)), ' Total'], 'Interpreter', 'None');
    elseif spHist==2
        xlabel('\DeltaY (P-A)')
    elseif spHist==3
        xlabel('\DeltaZ (V-D)')
        legend('General', 'Outlier Units');
    end
    
    set(gca, 'fontsize', 12, 'box', 'off')
    ylabel('Probability')
    
    %Included shaded area
    LeftEnd(1) = prctile(Sample1(:,spHist), 20);
    RightEnd(1) = prctile(Sample1(:,spHist), 80);
    LeftEnd(2) = prctile(Sample2(:,spHist), 20);
    RightEnd(2) = prctile(Sample2(:,spHist), 80);
    
    Yl = get(gca, 'ylim');
    rectangle('Position', [LeftEnd(1), 0, (RightEnd(1)-LeftEnd(1)), 0.9*Yl(2)],...
        'linewidth', 1, 'EdgeColor', [0.02, 0.31, 0.51, 0.2], 'FaceColor', [0.02, 0.31, 0.51, 0.2])%'FaceAlpha' is determined by the 4th #
    rectangle('Position', [LeftEnd(2), 0, (RightEnd(2)-LeftEnd(2)), 1.15*Yl(2)],...
        'linewidth', 1, 'EdgeColor', [1, 0.53, 0, 0.2], 'FaceColor', [1, 0.53, 0, 0.2])
    set(gca, 'ylim', [0, 1.15*Yl(2)])
    
end
saveas(fig, append(save_path, 'FR_', BrainRegion, '_xyz_hist.png'));

% Spike amp and duration histograms:
region_amps = T.amp(Neur_idx);
region_p2t = T.p2t(Neur_idx);
%region_p2t(region_p2t<0)=nan; %can use this if we want to compare only + spike widths
SpikeWF = [region_amps, region_p2t];

fig = figure;%(3)
for spHist=4:5
    subplot(1,2,spHist-3)
    Sample1(:,spHist) = SpikeWF(UnitsRegular, spHist-3);
    Sample2(:,spHist) = SpikeWF(UnitsOutliers, spHist-3);
    
    h1=histogram(Sample1(:,spHist),...
        'edgecolor', [0.02, 0.31, 0.51],...
        'normalization', 'probability', 'DisplayStyle', 'stairs', 'linewidth',2, 'edgealpha', 0.8); hold on;
    hold on
    h2=histogram(Sample2(:,spHist),...
        'normalization', 'probability',...
        'binwidth', get(h1, 'binwidth'), 'edgecolor', [1, 0.53, 0],...
        'facealpha', 0.4, 'edgealpha', 0.6, 'DisplayStyle', 'stairs', 'linewidth',3);
  
%     if max(Sample1(:,spHist)) < max(Sample2(:,spHist))
%         disp('error: fix plotting axes') %if 'error', fix plotting
%         break
%     end
    
    if spHist==4
        xlabel('WF Amp.')
        title([char(BrainRegion), ': ', num2str(sum(UnitsOutliers)),...
            ' Outlier neurons from ',...
            num2str(size(dXYZ,1)), ' Total'], 'Interpreter', 'None');
    elseif spHist==5
        xlabel('WF duration')
        legend('General', 'Outlier Units');
    end
    
    set(gca, 'fontsize', 12, 'box', 'off')
    ylabel('Probability')
    
end
saveas(fig, append(save_path, 'FR_', BrainRegion, '_amp_dur_hist.png'));

% Testing the signficance:
for st = 1:size(Sample1,2)
    [pMWU(st),hMWU(st)] = ranksum(Sample1(:,st), Sample2(:,st));
end
disp({'x','y','z','amp', 'dur'})
pMWU_corrected = pMWU*size(Sample1,2);
disp(pMWU_corrected) %Bonferroni correction (since 5 features are being compared: x,y,x,amp,duration)

%% Linear Regression Model: quantify the level of Avg FR variability explained by x,y,z,amp,p2t dur

% Covariates:
Covariates = [dXYZ(:,1), dXYZ(:,2), dXYZ(:,3), region_amps, region_p2t];
% To include lab ID as a covariate, too:
%Covariates = [cell2mat(Xloc)', cell2mat(Yloc)', cell2mat(Zloc)', cell2mat(amps)', cell2mat(peak_trough)', double(cell2mat(LabID))'];


% Run the model for shuffled data, as the control:
Output_shuffle = AvgFR(randperm(length(AvgFR)));
mdl_shuffle = fitlm(Covariates, Output_shuffle);
% mdl_shuffle.Coefficients %Shows which covariates have a significant weight
% R2_shuffle = mdl_shuffle.Rsquared; 
% %The R-squared value: the level of variability explained by the model.
% %Coefficient of determination and adjusted coefficient of determination, respectively.
% disp(R2_shuffle)

% Run the model for actual data (not shuffled):
Output = AvgFR; %(randperm(length(AvgFR))))';
mdl = fitlm(Covariates, Output);
% mdl.Coefficients %Shows which covariates have a significant weight
% R2 = mdl.Rsquared; 
% %The R-squared value: the level of variability explained by the model.
% %Coefficient of determination and adjusted coefficient of determination, respectively.
% disp(R2)

% To examine more model info:
%anova(mdl,'summary')
%figure; plot(mdl)
%figure;plot(mdl.Residuals.Raw, 'ok')
%figure;plot(Covariates(:,1), mdl.Residuals.Raw, 'ok')

%% Laminar analysis: plotting AvgFR or FF along depth (z axis) only
% % AvgFR
% figure
% swarmchart(ones(size(dXYZ,1), 1), dXYZ(:,3), 12, log10(AvgFR), 'filled')
% colormap parula
% caxis manual
% hCB=colorbar;
% hCB.Title.String = {'      Avg FR'; '      (spikes/sec)'};
% hCB.Title.FontSize = 12; %16;
% FRticks = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]; %to log10
% caxis([-0.9, 1.8]) %([log10(min(AvgFR)), 1.8])%log10(max(AvgFR))])
% % To include tick for threshold, but no label:
% hCB.Ticks = log10(sort([FRticks, FRthresh]));
% % To exclude the label for the threshold value: 
% TickLabelsTemp = 10.^(hCB.Ticks);
% TickLabelsCell = num2cell(TickLabelsTemp);
% FRthresh_idx = find(abs(10.^(hCB.Ticks) - FRthresh) == min(abs(10.^(hCB.Ticks) - FRthresh)));
% TickLabelsCell{FRthresh_idx} = [];
% set(hCB,'TickLabels', TickLabelsCell)
% ylabel('\DeltaZ (\mum)')
% xlim([-0.5, 2.5])
% set(gca, 'fontsize', 14, 'xtick', 1, 'xticklabel', [])% 'FR (sp/sec)')
% title(BrainRegion)
% 
% 
% %For FF:
% FFvals = T.avg_ff_post_move(Neur_idx);
% figure
% swarmchart(ones(size(dXYZ,1), 1), dXYZ(:,3), 12, FFvals, 'filled')
% colormap parula
% caxis manual
% caxis([0 3])
% hCB=colorbar;
% hCB.Title.String = {'FF post movement'};
% hCB.Title.FontSize = 12; %16;
% xlim([-0.5, 2.5])
% set(gca, 'fontsize', 14, 'xtick', 1, 'xticklabel', [])% 'FF')
% ylabel('\DeltaZ (\mum)')
% title(BrainRegion)
% stop=1;


function [] = Fig3Dplots_generate(CSVfile, BrainRegion, save_path)

TM_test1 = 'start_to_move'; %'post_stim'; %'trial'; %'post_reward';%
TM_test2 = 'pre_move_lr'; %'pre_move'; %'post_move'; %

% Sets the 3D plot's ylabel rotation depending on brain region view rotation:
if strcmp(BrainRegion, 'PPC')
    rot = -25; pos1 = [-687  -346  -728]; pos2=[-697  -391  -729]; pos3=[-657  -360  -744];
elseif strcmp(BrainRegion, 'CA1')
    rot = -30; pos1 = [-605  -374  -655]; pos2=[-603  -432  -669]; pos3=[-569  -403  -695];
elseif strcmp(BrainRegion, 'DG')
    rot = -30; pos1 = [-605  -327  -507]; pos2=[-596  -389  -526]; pos3=[-596  -389  -526];
elseif strcmp(BrainRegion, 'LP')
    rot = -40; pos1 = [-642 -343 -612]; pos2=[-667 -380 -600]; pos3=[-650 -338 -609];
elseif strcmp(BrainRegion, 'PO')
    rot = -33; pos1 = [-772  -317  -698]; pos2=[-779  -357  -672]; pos3=[-755  -320  -688];
end

f = figure(1);
ax(1) = subplot(3,2,2);
[hCB, jitt, FRthresh, pMWU_corrFR, mdlFR, mdlFR_shuffle,...
    FRthreshLow] = plot3D_FR(CSVfile,...
    BrainRegion, ax(1), save_path);
set(hCB, 'position', [0.9326  0.7002  0.0085  0.2355]) %[0.9175  0.6879  0.0085  0.2355]) %[0.9175  0.6930  0.0201  0.2157])
set(get(ax(1), 'Ylabel'), 'Rotation', rot, 'Position', pos1)

figure(f)
ax(2) = subplot(3,2,4);
[hLegend1, pMWU_corrTM1, mdlTM1, mdlTM1_shuffle] = plot3D_TM(CSVfile,...
    BrainRegion, TM_test1, jitt, ax(2), save_path);
set(hLegend1, 'box', 'on', 'Position', [0.7310  0.4277  0.1713  0.0363])
set(get(ax(2), 'Ylabel'), 'Rotation', rot, 'Position', pos2)

figure(f)
ax(3) = subplot(3,2,6);
[hLegend2, pMWU_corrTM2, mdlTM2, mdlTM2_shuffle] = plot3D_TM(CSVfile,...
    BrainRegion, TM_test2, jitt, ax(3), save_path);
set(hLegend2, 'box', 'on', 'Position', [0.7445  0.1285  0.1619 0.0342])
set(get(ax(3), 'Ylabel'), 'Rotation', rot, 'Position', pos3)

figure(f)
ax(4) = subplot(3,2,3);
[hLegend3] = plot_FRhist_TM(CSVfile, BrainRegion, TM_test1);
%set(hLegend2, 'box', 'off', 'Position', [0.7445  0.1285  0.1619 0.0342])
xlabel('\Delta FR (TW pre-movement - pre-stim)')
set(ax(4), 'fontsize',12)

figure(f)
ax(5) = subplot(3,2,5);
[hLegend4] = plot_FRhist_TM(CSVfile, BrainRegion, TM_test2);
%set(hLegend2, 'box', 'off', 'Position', [0.7445  0.1285  0.1619 0.0342])
xlabel('\Delta FR (Rstim - Lstim)')
set(ax(5), 'fontsize',12)

figure(f)
f.Position = [996 38 797 979];

%% Go to subplot 2: Annotate the colorbar to separate regular and outlier neurons:

% Find position of colobar and its min/max values:
barPos = get(hCB, 'Position');
cLimits = caxis(ax(1));
% Find vertical position of FRthresh on colorbar:
PositionRatio = (log10(FRthresh)-cLimits(1))/diff(cLimits);
YlineLoc = barPos(2) + barPos(4)*PositionRatio;
% Select horizontal position of line, centered on colorbar, and plot line:
XlineLoc = barPos(1) + barPos(3)/2 + [0.014, -0.014];
%XlineLoc = barPos(1) + barPos(3)/2 + [0.022, -0.022];
h_colorbarLine = annotation('line', XlineLoc, [YlineLoc, YlineLoc], 'Color', [0.9 0.6 0.2], 'linewidth', 2.5);                
if ~isnan(FRthreshLow)
    PositionRatioLow = (log10(FRthreshLow)-cLimits(1))/diff(cLimits);
    YlineLocLowFR = barPos(2) + barPos(4)*PositionRatioLow;
    h_colorbarLineLowFR = annotation('line', XlineLoc,...
        [YlineLocLowFR, YlineLocLowFR], 'Color', [0.3 0.4 0.9], 'linewidth', 2.5);                
end

%% Linear Regression Model analysis:
% Linear regression model for shuffled data (as the control):
mdlFR_shuffle.Coefficients %Shows which covariates have a significant weight
R2_shuffle = mdlFR_shuffle.Rsquared; 
disp(R2_shuffle)

% Linear regression model for actual data (not shuffled):
mdlFR.Coefficients %Shows which covariates have a significant weight
R2 = mdlFR.Rsquared; 
%The R-squared value: the level of variability explained by the model.
%Coefficient of determination and adjusted coefficient of determination, respectively.
disp(R2)

%% Display signif. difference in x, y, z, WF amp, and WF p2t duration:
disp({'x','y','z','amp','p2t dur'})
disp([pMWU_corrFR; pMWU_corrTM1; pMWU_corrTM2])

saveas(f, append(save_path, 'figure7_', BrainRegion, '.png'))

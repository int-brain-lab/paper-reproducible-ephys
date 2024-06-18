% Code for generating 3D plots of task-modulated neurons and examining
% spatial position and spike waveform characteristics as a source of
% variability (e.g., in Fig. 7c-d).
% Code written by: Marsa Taheri

function [hLegend, pMWU_corrected,...
    mdl, mdl_shuffle] = plot3D_TM(CSVfile, BrainRegion, TM_test, jitt, ax, save_path)

%mdl and mdl_shuffle are the linear regression model for real and shuffled
%data
%jitt is the xyz jitter/noise that is added to each data point, used
%previously in plot3D_FR.m and output from there (for consistency between
%plots).

T = readtable(CSVfile);

X_col = find(strcmp(T.Properties.VariableNames, 'x'));
Y_col = find(strcmp(T.Properties.VariableNames, 'y'));
Z_col = find(strcmp(T.Properties.VariableNames, 'z'));

%Find neurons in specified brain region:
%Neur_idx = find(strcmp(T.region, BrainRegion));
%If we want to use only the permut_include pids from the Task Modulation fig:
Neur_idx = find(strcmp(T.region, BrainRegion) & strcmp(T.permute_include, 'True'));
%For a specific lab:
%Neur_idx = find(strcmp(T.lab, 'churchlandlab_ucla') & strcmp(T.region, BrainRegion));
%Neur_idx = find(strcmp(T.lab, 'mainenlab') & strcmp(T.region, BrainRegion));


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Code below is for when using Fig_task_modulation data, instead of Fig6 data
% % and we need to only include neurons from recordings that are included in 
% %the permutation. First, check to make sure order matches the python cvs order:
% 
% %T_permut = readtable('/Users/mt/int-brain-lab/paper-reproducible-ephys/FigTM_PermutInc_dataframe_DG.csv');
% T_permut = readtable('/Users/mt/int-brain-lab/paper-reproducible-ephys/FigTM_PermutInc_dataframe_PO.csv');
% idx_clust = T.cluster_ids(Neur_idx);
% Neur_idx_new = [];
% if sum(idx_clust == T_permut.cluster_ids) == length(Neur_idx)
%     for j = 1:length(Neur_idx)
%         if strcmp(cell2mat(T_permut.permute_include(j)), 'True')==1
%            Neur_idx_new = [Neur_idx_new; Neur_idx(j)];
%         end
%     end
%     Neur_idx = Neur_idx_new;
% else
%     display('ERROR: DATAFRAME DOESN''T MATCH THE REGION CSV FILE')
% end
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
%jittSize = ones(1,3)*min([range(abs(dXYZ(:,1))), range(abs(dXYZ(:,2))), range(abs(dXYZ(:,3)))])./70;
%jitt = jittSize.*(rand(size(dXYZ(:,1), 1), 3)-0.5); %center the noise around 0 (+ and -) with - 0.5

dXjitt = dXYZ(:,1) + jitt(:,1);
dYjitt = dXYZ(:,2) + jitt(:,2);
dZjitt = dXYZ(:,3) + jitt(:,3);

%% Find neurons that are task modulated (TM) and non-modulated (NM) with specified test.
% Then, find x,y,z position of those 2 groups of neurons.

%TM_test = 'start_to_move'; %'pre_move_lr'; %'start_to_move';
switch TM_test
    case 'start_to_move' %strcmp(TMtest, 'start_to_move')==1
        pVal_neur = T.p_start_to_move(Neur_idx);
        p_TF_neur = T.start_to_move(Neur_idx);
    case 'pre_move_lr'
        pVal_neur = T.p_pre_move_lr(Neur_idx);
        p_TF_neur = T.pre_move_lr(Neur_idx);
    case 'post_stim'
        pVal_neur = T.p_post_stim(Neur_idx);
        p_TF_neur = T.post_stim(Neur_idx);
    case 'pre_move'
        pVal_neur = T.p_pre_move(Neur_idx);
        p_TF_neur = T.pre_move(Neur_idx);
    case 'trial'
        pVal_neur = T.p_trial(Neur_idx);
        p_TF_neur = T.trial(Neur_idx);
    case 'post_move'
        pVal_neur = T.p_post_move(Neur_idx);
        p_TF_neur = T.post_move(Neur_idx);
    case 'post_reward'
        pVal_neur = T.p_post_reward(Neur_idx);
        p_TF_neur = T.post_reward(Neur_idx);
end

%Convert vector with True or False strings to a logical vector:
p_logic_neur = strcmpi(p_TF_neur, 'true');

TM_scatter = p_logic_neur(p_logic_neur);
NM_scatter = p_logic_neur(p_logic_neur==0);

dXjitt_TM = dXjitt(p_logic_neur);
dYjitt_TM = dYjitt(p_logic_neur);
dZjitt_TM = dZjitt(p_logic_neur);
dXjitt_NonM = dXjitt(p_logic_neur==0);
dYjitt_NonM = dYjitt(p_logic_neur==0);
dZjitt_NonM = dZjitt(p_logic_neur==0);


%% Make 3D plot
%figure(1)
SmallDotSz = 10; LargeDotSz = 45; FontSz=12;

scatter3(dXjitt_TM, dYjitt_TM, dZjitt_TM, LargeDotSz, TM_scatter,...
    'filled', 'MarkerFaceAlpha', 0.7,...
    'markeredgecolor', [1 1 1], 'markeredgealpha', 0.9, 'linewidth', 0.8)
hold on
scatter3(dXjitt_NonM, dYjitt_NonM, dZjitt_NonM, SmallDotSz, NM_scatter,...
    'filled', 'MarkerFaceAlpha', 0.7,...
    'markeredgecolor', [1 1 1], 'markeredgealpha', 0.9, 'linewidth', 0.8)

XtickAuto = get(gca, 'Xtick'); YtickAuto = get(gca, 'Ytick'); ZtickAuto = get(gca, 'Ztick');
set(gca, 'fontsize', FontSz, 'Xtick', min(XtickAuto):200:max(XtickAuto),...
    'Ytick', min(YtickAuto):200:max(YtickAuto), 'Ztick', min(ZtickAuto):200:max(ZtickAuto))
%colormap copper
cmap = [0.5 0.5 0.5; 0.9, 0.6, 0.4]; %[0.5 0.5 0.5; 1, 0.41, 0.16];
colormap(ax, cmap)
%To binarize an existing colormap:
%colormap copper; cmap = colormap; cmap2 = [cmap(1,:); cmap(end,:)]; colormap(cmap2)
%caxis manual; %caxis([1 2.4]); %colorbar

hold on %plotting the target center of mass
sc=scatter3(0,0,0, 'x', 'linewidth', 2, 'markeredgecolor', [0.8 0.2 0.2]);
sc.SizeData=70;
sc=scatter3(0,0,0, 'or', 'markerfacecolor', 'r', 'markerfacealpha', 0.25);
sc.SizeData=55;
axis tight

xlabel('\DeltaX (\mum)'); ylabel('\DeltaY (\mum)'); zlabel('\DeltaZ (\mum)')
%title(BrainRegion)
hLegend = legend('Task-Modulated', 'Not Modulated', 'fontsize', 12);
%set(hLegend, 'Position', [0.5732    0.2262    0.2830    0.1012], 'Units', 'points')

% xlim([X_lim(1), X_lim(2)])
% ylim([Y_lim(1), Y_lim(2)])
% zlim([Z_lim(1), Z_lim(2)])

% Sets the azimuth and elevation angles of the view:
if strcmp(BrainRegion, 'PPC')
    view(-25,13)
elseif strcmp(BrainRegion, 'CA1')
    view(-19, 13)
elseif strcmp(BrainRegion, 'DG')
    view(-18,12)
elseif strcmp(BrainRegion, 'LP')
    view(-13, 21) %view(-17,24)
elseif strcmp(BrainRegion, 'PO')
    view(-24,18)
end
%To get azimuth and elevation of the current view: [caz,cel] = view()
hold off


%% Separate the two groups, then statistics (what's unique about outlier clusters):

% x, y, z histograms:
fig = figure; %(2)
for spHist=1:3
    subplot(3,1,spHist)
    Sample1(:,spHist) = dXYZ(p_logic_neur, spHist);
    Sample2(:,spHist) = dXYZ(p_logic_neur==0, spHist);
    
    h1=histogram(Sample1(:,spHist),...
        'binwidth',100, 'edgecolor', [1, 0.41, 0.16],...
        'normalization', 'probability', 'DisplayStyle', 'stairs', 'linewidth',2, 'edgealpha', 0.8); hold on;
    hold on
    h2=histogram(Sample2(:,spHist),...
        'normalization', 'probability',...
        'binwidth',100, 'edgecolor', [0.5, 0.5, 0.5],...
        'facealpha', 0.4, 'edgealpha', 0.6, 'DisplayStyle', 'stairs', 'linewidth',3);
    
    if spHist==1
        xlabel('\DeltaX (L-M)')
        title([char(BrainRegion), ', ', TM_test, ': ', num2str(sum(p_logic_neur)),...
            ' TM neurons from ',...
            num2str(size(dXYZ,1)), ' Total'],'Interpreter','none');
        %title([char(BrainRegion), ', ', TM_test, ': ', num2str(sum(p_logic_neur)),' TM neurons from ',num2str(size(dXYZ,1)), ' Total'], 'Interpreter','none')
    elseif spHist==2
        xlabel('\DeltaY (P-A)')
    elseif spHist==3
        xlabel('\DeltaZ (V-D)')
        legend('Task-Modulated', 'Non-Modulated');
    end
    
    set(gca, 'fontsize', 12, 'box', 'off')
    ylabel('Probability')
    
    %Included shaded area
    LeftEnd(1) = prctile(Sample1(:,spHist), 20);
    RightEnd(1) = prctile(Sample1(:,spHist), 80);
    LeftEnd(2) = prctile(Sample2(:,spHist), 20);
    RightEnd(2) = prctile(Sample2(:,spHist), 80);
    
    Yl = get(gca, 'ylim');
    rectangle('Position', [LeftEnd(1), 0, (RightEnd(1)-LeftEnd(1)), 1.15*Yl(2)],...
        'linewidth', 1, 'EdgeColor', [1, 0.41, 0.16, 0.2], 'FaceColor', [1, 0.41, 0.16, 0.2])%'FaceAlpha' is determined by the 4th #
    rectangle('Position', [LeftEnd(2), 0, (RightEnd(2)-LeftEnd(2)), 0.9*Yl(2)],...
        'linewidth', 1, 'EdgeColor', [0.5, 0.5, 0.5, 0.2], 'FaceColor', [0.5, 0.5, 0.5, 0.2])
    set(gca, 'ylim', [0, 1.15*Yl(2)])
    
end

saveas(fig, append(save_path, TM_test, BrainRegion, '_xyz_hist.png'));

% Spike amp and duration histograms:
region_amps = T.amp(Neur_idx);
region_p2t = T.p2t(Neur_idx);
% %Remove negative spike widths from analysis:
% region_p2t(region_p2t<0) = nan;

SpikeWF = [region_amps, region_p2t];

fig = figure; %(3)
for spHist=4:5
    subplot(1,2,spHist-3)
    Sample1(:,spHist) = SpikeWF(p_logic_neur, spHist-3);
    Sample2(:,spHist) = SpikeWF(p_logic_neur==0, spHist-3);
    
    h1=histogram(Sample1(:,spHist),...
        'edgecolor', [1, 0.41, 0.16],...
        'normalization', 'probability', 'DisplayStyle', 'stairs', 'linewidth',2, 'edgealpha', 0.8); hold on;
    hold on
    h2=histogram(Sample2(:,spHist),...
        'normalization', 'probability',...
        'binwidth', get(h1, 'binwidth'), 'edgecolor', [0.5, 0.5, 0.5],...
        'facealpha', 0.4, 'edgealpha', 0.6, 'DisplayStyle', 'stairs', 'linewidth',3);

    
    if spHist==4
        xlabel('WF Amp.')
        title([char(BrainRegion), ', ', TM_test, ': ', num2str(sum(p_logic_neur)),...
            ' TM neurons from ',...
            num2str(size(dXYZ,1)), ' Total'],'Interpreter','none');
    elseif spHist==5
        xlabel('WF duration')
        legend('Task-Modulated', 'Non-Modulated');
    end
    
    set(gca, 'fontsize', 12, 'box', 'off')
    ylabel('Probability')
    
end
saveas(fig, append(save_path, TM_test, BrainRegion, '_amp_dur_hist.png'));

% Testing the signficance:
for st = 1:size(Sample1,2)
    [pMWU(st),hMWU(st)] = ranksum(Sample1(:,st), Sample2(:,st));
end
%disp({'x','y','z','amp', 'dur'})
pMWU_corrected = pMWU*size(Sample1,2);


%% Linear Regression Model: quantify the level of Avg FR variability explained by x,y,z,amp,p2t dur

% Covariates:
Covariates = [dXYZ(:,1), dXYZ(:,2), dXYZ(:,3), region_amps, region_p2t];
% To include lab ID as a covariate, too:
%Covariates = [cell2mat(Xloc)', cell2mat(Yloc)', cell2mat(Zloc)', cell2mat(amps)', cell2mat(peak_trough)', double(cell2mat(LabID))'];


% Run the model for shuffled data, as the control:
Output_shuffle = double(p_logic_neur(randperm(length(p_logic_neur))));
mdl_shuffle = fitlm(Covariates, Output_shuffle);
% mdl_shuffle.Coefficients %Shows which covariates have a significant weight
% R2_shuffle = mdl_shuffle.Rsquared;
% %The R-squared value: the level of variability explained by the model.
% %Coefficient of determination and adjusted coefficient of determination, respectively.
% disp(R2_shuffle)

% Run the model for actual data (not shuffled):
Output = double(p_logic_neur); %(randperm(length(AvgFR))))';
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


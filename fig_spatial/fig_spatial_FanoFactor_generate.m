% Code for generating Fano Factor 3D plots
% Code written by: Marsa Taheri

function [] = fig_spatial_FanoFactor_generate(CSVfile, BrainRegions, save_path)

pMWU_corrFR_all = nan(5, 5); %5 brain regions possible & 5features being compared (x,y,z,amp,dur)
f = figure(1);
f.Position = [1431   37   362  979];
%amps_all=[];

for regionN = 1:size(BrainRegions,2)
    clear Sample1 Sample2
    BrainRegion = BrainRegions(regionN);
    % Sets the 3D plot's ylabel rotation depending on brain region view rotation:
    if strcmp(BrainRegion, 'PPC')
        sp=1; rot = -25; pos1 = [-711  -415  -756];
    elseif strcmp(BrainRegion, 'CA1')
        sp=2; rot = -30; pos1 = [-580  -482  -756];
    elseif strcmp(BrainRegion, 'DG')
        sp=3; rot = -30; pos1 = [-619  -393  -551];
    elseif strcmp(BrainRegion, 'LP')
        sp=4; rot = -41; pos1 = [-673  -393  -703];
    elseif strcmp(BrainRegion, 'PO')
        sp=5; rot = -25; pos1 = [-795  -389  -775];
    end
      
    %ax = subplot(5,1,sp);
    T = readtable(CSVfile);
    
    %AvgFR_col = find(strcmp(T.Properties.VariableNames, 'avg_fr'));
    X_col = find(strcmp(T.Properties.VariableNames, 'x'));
    Y_col = find(strcmp(T.Properties.VariableNames, 'y'));
    Z_col = find(strcmp(T.Properties.VariableNames, 'z'));
    
    %Find neurons in specified brain region that also have an avg FR >FRthresh:
    FFmax = 3; %later for plotting purposes, to set max displayed FF
    FRthresh = 1; % min avg FR in order to calculate proper FF
    AvgFR = T.avg_fr;
    Neur_idx = find(strcmp(T.region, BrainRegion) & AvgFR>=FRthresh);
    
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
    figure(f)
    ax = subplot(5,1,sp);
    SmallDotSz = 16; LargeDotSz = SmallDotSz*2.5; FontSz=10;
    
    FFvalues = T.avg_ff_post_move(Neur_idx);
    
    % Identify neurons with high and low FFs post movement:
    UnitsHighFF = FFvalues>=1;
    UnitsLowFF = FFvalues<1;
    
    FF_High = FFvalues(UnitsHighFF);
    %For plotting purposes, set any neuron w/FF>3 to FFmax; indicate on colorbar:
    FF_High(FF_High>FFmax)=FFmax;
    dX_HighFF = dXjitt(UnitsHighFF);
    dY_HighFF = dYjitt(UnitsHighFF);
    dZ_HighFF = dZjitt(UnitsHighFF);
    scatter3(dX_HighFF, dY_HighFF, dZ_HighFF, SmallDotSz, FF_High,...
        'filled', 'MarkerFaceAlpha', 0.6,...
        'markeredgecolor', [1 1 1], 'markeredgealpha', 0.9, 'linewidth', 0.8)
    
    hold on %plotting the target center of mass
    sc=scatter3(0,0,0, 'x', 'linewidth', 2, 'markeredgecolor', [0.8 0.2 0.2]);
    sc.SizeData=70;
    sc=scatter3(0,0,0, 'or', 'markerfacecolor', 'r', 'markerfacealpha', 0.25);
    sc.SizeData=55;
    axis tight
    
    FF_Low = FFvalues(UnitsLowFF);
    dX_outlier = dXjitt(UnitsLowFF);
    dY_outlier = dYjitt(UnitsLowFF);
    dZ_outlier = dZjitt(UnitsLowFF);
    
    scatter3(dX_outlier, dY_outlier, dZ_outlier, LargeDotSz, FF_Low','filled',...
        'markerfacealpha', 0.7,'markeredgecolor','k', 'linewidth', 0.8);
    % Set x, y, z ticks:
    XtickAuto = get(gca, 'Xtick'); YtickAuto = get(gca, 'Ytick'); ZtickAuto = get(gca, 'Ztick');
    set(gca, 'Xtick', min(XtickAuto):200:max(XtickAuto),...
        'Ytick', min(YtickAuto):200:max(YtickAuto), 'Ztick', min(ZtickAuto):200:max(ZtickAuto))
    
    colormap parula
    caxis manual
    hCB=colorbar;
    hCB.Title.String = 'Fano Factor';
    hCB.Title.FontSize = FontSz;%12; %16;
    xlabel('\DeltaX (\mum)'); ylabel('\DeltaY (\mum)'); zlabel('\DeltaZ (\mum)')
    set(gca, 'fontsize', FontSz)
    caxis([min(FFvalues), FFmax]) %([0, FFmax])%
    
    %set max of ticklabel to FF max thresh:
    tick_lab = get(hCB, 'ticklabels');
    tick_lab(end) = {['>',num2str(FFmax)]};
    set(hCB, 'ticklabels', tick_lab)
    
    %set(hCB, 'position', [0.9326  0.7002  0.0085  0.2355]) %[0.9175  0.6879  0.0085  0.2355]) %[0.9175  0.6930  0.0201  0.2157])
        
    % Annotate the colorbar to separate regular and outlier neurons:
    % Find position of colobar and its min/max values:
    barPos = get(hCB, 'Position');
    ax_pos = get(ax, 'Position');
    set(hCB, 'Position', [barPos(1)*1.08,barPos(2),barPos(3)*0.7,barPos(4)*1.2])
    barPosNew = get(hCB, 'Position');
    %Due to stretching of original 3D subplot, reset its position:
    %set(ax, 'Position', ax_pos) 
    set(ax, 'Position', [ax_pos(1)*1.12, ax_pos(2), ax_pos(3)*1.1, ax_pos(4)*1.2])
    %if sp==1
    %    set(ax, 'Position', [ax_pos(1)*1.3, ax_pos(2)*1.05, ax_pos(3), ax_pos(4)*1.1])
    %else        
    %    set(ax, 'Position', [ax_pos(1)*1.3, ax_pos(2)*0.98, ax_pos(3), ax_pos(4)*1.1])
    %end
    cLimits = caxis();
    % Find vertical position of FRthresh on colorbar:
    PositionRatio = (1 - cLimits(1))/diff(cLimits);
    YlineLoc = barPosNew(2) + barPosNew(4)*PositionRatio;
    % Select horizontal position of line, centered on colorbar, and plot line:
    XlineLoc = barPosNew(1) + barPosNew(3)/2 + [0.03, -0.03];
    h_colorbarLine = annotation('line', XlineLoc, [YlineLoc, YlineLoc],...
        'Color', [0.3 0.4 0.9], 'linewidth', 2.5);
    
    
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
    set(get(ax, 'Ylabel'), 'Rotation', rot, 'Position', pos1)
    
    %% Separate the two groups, then statistics (what's unique about outlier clusters):
    
    % x, y, z histograms:
    fig = figure;%(2)
    for spHist=1:3
        subplot(3,1,spHist)
        Sample1(:,spHist) = dXYZ(UnitsLowFF, spHist);
        Sample2(:,spHist) = dXYZ(UnitsHighFF, spHist);
        
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
            title([char(BrainRegion), ': ', num2str(sum(UnitsLowFF)),...
                ' Outlier neurons from ',...
                num2str(size(dXYZ,1)), ' Total'], 'Interpreter', 'None');
        elseif spHist==2
            xlabel('\DeltaY (P-A)')
        elseif spHist==3
            xlabel('\DeltaZ (V-D)')
            legend('FF < 1','FF > 1')%('Regular Units', 'Outlier Units');
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
    saveas(fig, append(save_path, 'FF_', BrainRegion, '_xyz_hist.png'));
    
    % Spike amp and duration histograms:
    region_amps = T.amp(Neur_idx);
    region_p2t = T.p2t(Neur_idx);
    %Remove negative spike widths from analysis
    region_p2t(region_p2t<0) = nan;
    
    SpikeWF = [region_amps, region_p2t];
    %amps_all = [amps_all, region_amps'];
    
    fig = figure;%(3)
    for spHist=4:5
        subplot(1,2,spHist-3)
        Sample1(:,spHist) = SpikeWF(UnitsLowFF, spHist-3);
        Sample2(:,spHist) = SpikeWF(UnitsHighFF, spHist-3);
        
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
            title([char(BrainRegion), ': ', num2str(sum(UnitsLowFF)),...
                ' Outlier neurons from ',...
                num2str(size(dXYZ,1)), ' Total'], 'Interpreter', 'None');
        elseif spHist==5
            xlabel('WF duration')
            legend('FF < 1','FF > 1')%('Regular Units', 'Outlier Units');
        end
        
        set(gca, 'fontsize', 12, 'box', 'off')
        ylabel('Probability')
        
    end
    saveas(fig, append(save_path, 'FF_', BrainRegion, '_amp_dur_hist.png'));
    
    % Testing the signficance:
    for st = 1:size(Sample1,2)
        [pMWU(st),hMWU(st)] = ranksum(Sample1(:,st), Sample2(:,st));
    end
    pMWU_corrected = pMWU*size(Sample1,2);
    pMWU_corrFR_all(sp,:) = pMWU_corrected;
    
    %% Linear Regression Model: quantify the level of Avg FR variability explained by x,y,z,amp,p2t dur
    
    % Covariates:
    Covariates = [dXYZ(:,1), dXYZ(:,2), dXYZ(:,3), region_amps, region_p2t];
    % To include lab ID as a covariate, too:
    %Covariates = [cell2mat(Xloc)', cell2mat(Yloc)', cell2mat(Zloc)', cell2mat(amps)', cell2mat(peak_trough)', double(cell2mat(LabID))'];
    
    
    % Run the model for shuffled data, as the control:
    Output_shuffle = FFvalues(randperm(length(FFvalues)));
    mdl_shuffle = fitlm(Covariates, Output_shuffle);
    % mdl_shuffle.Coefficients %Shows which covariates have a significant weight
    % R2_shuffle = mdl_shuffle.Rsquared;
    % %The R-squared value: the level of variability explained by the model.
    % %Coefficient of determination and adjusted coefficient of determination, respectively.
    % disp(R2_shuffle)
    
    % Run the model for actual data (not shuffled):
    Output = FFvalues; %(randperm(length(AvgFR))))';
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
    
    
    %% Scatter plot of FF vs. width with marker size from avg FR
    %Get rid of nan FFs and negative spike widths/durations:
    x_dur = SpikeWF(~isnan(FFvalues),2);
    y_FF = FFvalues(~isnan(FFvalues));
    z_FR = AvgFR(Neur_idx);
    z_FR = z_FR(~isnan(FFvalues));
    y_FF = y_FF(x_dur>0);
    z_FR = z_FR(x_dur>0);
    x_dur = x_dur(x_dur>0);
    %Get rid of very high FFs:
    x_dur = x_dur(y_FF<=3);
    z_FR = z_FR(y_FF<=3);
    y_FF = y_FF(y_FF<=3);
    
    figure
    scatter(x_dur, y_FF, z_FR*2, 'o')

    %Fit line to data using polyfit
    c = polyfit(x_dur, y_FF, 1);%(x,y,1);
    %%Display evaluated equation y = m*x + b
    %disp(['Equation is y = ' num2str(c(1)) '*x + ' num2str(c(2))])

    % Evaluate fit equation using polyval
    y_est = polyval(c, x_dur);
    % Add trend line to plot
    hold on
    plot(x_dur, y_est,'r','LineWidth',1)
    
    [Rcc,Pcc] = corrcoef(x_dur, y_FF);
    disp([Rcc(1,2)^2, Pcc(1,2)])
    
    set(gca, 'fontsize', 15)
    xlabel('Spike width (ms)')
    ylabel('Fano Factor post-movement')
    title(char(BrainRegion))
    
%     scatter(SpikeWF(:,2), FFvalues)
%     hold on
%     scatter(SpikeWF(UnitsLowFF,2), FFvalues(UnitsLowFF))
end

%% Display signif. difference in x, y, z, WF amp, and WF p2t duration:
disp({'x','y','z','amp','p2t dur'})
disp(pMWU_corrFR_all)
disp(pMWU_corrFR_all<0.05)
saveas(f, append(save_path, 'figure_spatial_FFsupp.png'))

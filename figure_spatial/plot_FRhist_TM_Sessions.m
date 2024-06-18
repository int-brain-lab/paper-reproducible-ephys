% Code for generating firing rate histograms of task-modulated neurons
% (e.g., in Fig. 7c-d, left side).
% Code written by: Marsa Taheri

function [hLegend] = plot_FRhist_TM_Sessions(CSVfile, BrainRegion, TM_test, modulated)

T = readtable(CSVfile);

%Find neurons in specified brain region:
Neur_idx_region = find(strcmp(T.region, BrainRegion));

Nlabs = length(unique(T.institute(Neur_idx_region)));
labs = unique(T.institute(Neur_idx_region));

%------ FF:
FF_region = T.avg_ff_post_move(Neur_idx_region);
%%writematrix(FF_region, ['tableFF_', char(BrainRegion),'.csv'])
pid = T.pid(Neur_idx_region);
cluster_ids = T.cluster_ids(Neur_idx_region);
institute = T.institute(Neur_idx_region);
Table_FF = table(pid, cluster_ids, institute, FF_region);
%writetable(Table_FF, ['table_FF_', char(BrainRegion),'.csv'])
%------


for n = 1:Nlabs
    
    Neur_idx = find(strcmp(T.region, BrainRegion) & strcmp(T.institute, labs{n}));
    
    %% Find neurons that are task modulated (TM) and non-modulated (NM) with specified test.
    
    %TM_test = 'start_to_move'; %'pre_move_lr'; %'start_to_move';
    switch TM_test
        case 'start_to_move' %strcmp(TMtest, 'start_to_move')==1
            %pVal_neur = T.p_start_to_move(Neur_idx);
            p_TF_neur = T.start_to_move(Neur_idx);
        case 'pre_move_lr'
            %pVal_neur = T.p_pre_move_lr(Neur_idx);
            p_TF_neur = T.pre_move_lr(Neur_idx);
        case 'post_stim'
            %pVal_neur = T.p_post_stim(Neur_idx);
            p_TF_neur = T.post_stim(Neur_idx);
        case 'pre_move'
            %pVal_neur = T.p_pre_move(Neur_idx);
            p_TF_neur = T.pre_move(Neur_idx);
        case 'trial'
            %pVal_neur = T.p_trial(Neur_idx);
            p_TF_neur = T.trial(Neur_idx);
        case 'post_move'
            %pVal_neur = T.p_post_move(Neur_idx);
            p_TF_neur = T.post_move(Neur_idx);
        case 'post_reward'
            %pVal_neur = T.p_post_reward(Neur_idx);
            p_TF_neur = T.post_reward(Neur_idx);
    end
    
    %Convert vector with True or False strings to a logical vector:
    p_logic_neur = strcmpi(p_TF_neur, 'true');
    
    avg_fr_base_reg = T.avg_fr_base(Neur_idx);
    switch TM_test
        case 'start_to_move' %strcmp(TMtest, 'start_to_move')==1
            avg_fr_pre_move_tw_reg = T.avg_fr_pre_move_tw(Neur_idx);
            FR0_TM = avg_fr_base_reg(p_logic_neur);
            FR1_TM = avg_fr_pre_move_tw_reg(p_logic_neur);
            FR0_NonM = avg_fr_base_reg(p_logic_neur==0);
            FR1_NonM = avg_fr_pre_move_tw_reg(p_logic_neur==0);
            TM_test_title = 'Reaction Period vs. Pre-stim Test';
        case 'pre_move_lr'
            avg_fr_pre_moveL_reg = T.avg_fr_pre_moveL(Neur_idx);
            avg_fr_pre_moveR_reg = T.avg_fr_pre_moveR(Neur_idx);
            FR0_TM = avg_fr_pre_moveL_reg(p_logic_neur);
            FR1_TM = avg_fr_pre_moveR_reg(p_logic_neur);
            FR0_NonM = avg_fr_pre_moveL_reg(p_logic_neur==0);
            FR1_NonM = avg_fr_pre_moveR_reg(p_logic_neur==0);
            TM_test_title = 'Left vs. Right Movement Test';
        case 'post_stim'
            avg_fr_post_stim_reg = T.avg_fr_post_stim(Neur_idx);
            FR0_TM = avg_fr_base_reg(p_logic_neur);
            FR1_TM = avg_fr_post_stim_reg(p_logic_neur);
            FR0_NonM = avg_fr_base_reg(p_logic_neur==0);
            FR1_NonM = avg_fr_post_stim_reg(p_logic_neur==0);
            TM_test_title = 'Stimulus Test';
        case 'pre_move'
            avg_fr_pre_move_reg = T.avg_fr_pre_move(Neur_idx);
            FR0_TM = avg_fr_base_reg(p_logic_neur);
            FR1_TM = avg_fr_pre_move_reg(p_logic_neur);
            FR0_NonM = avg_fr_base_reg(p_logic_neur==0);
            FR1_NonM = avg_fr_pre_move_reg(p_logic_neur==0);
            TM_test_title = 'Movement Initiation Test';
        case 'trial'
            avg_fr_trial_reg = T.avg_fr_trial(Neur_idx);
            FR0_TM = avg_fr_base_reg(p_logic_neur);
            FR1_TM = avg_fr_trial_reg(p_logic_neur);
            FR0_NonM = avg_fr_base_reg(p_logic_neur==0);
            FR1_NonM = avg_fr_trial_reg(p_logic_neur==0);
            TM_test_title = TM_test;
        case 'post_move'
            avg_fr_post_move_reg = T.avg_fr_post_move(Neur_idx);
            FR0_TM = avg_fr_base_reg(p_logic_neur);
            FR1_TM = avg_fr_post_move_reg(p_logic_neur);
            FR0_NonM = avg_fr_base_reg(p_logic_neur==0);
            FR1_NonM = avg_fr_post_move_reg(p_logic_neur==0);
            TM_test_title = 'Movement Test';
        case 'post_reward'
            avg_fr_post_reward_reg = T.avg_fr_post_reward(Neur_idx);
            FR0_TM = avg_fr_base_reg(p_logic_neur);
            FR1_TM = avg_fr_post_reward_reg(p_logic_neur);
            FR0_NonM = avg_fr_base_reg(p_logic_neur==0);
            FR1_NonM = avg_fr_post_reward_reg(p_logic_neur==0);
            TM_test_title = 'Reward Test';
    end
    
    DeltaFR_of_TM = FR1_TM - FR0_TM;
    DeltaFR_of_NonM = FR1_NonM - FR0_NonM;
    
    
    if modulated==1
        %         histogram(DeltaFR_of_TM, 'binwidth', 0.2, 'DisplayStyle', 'stairs',...
        %             'linewidth',1.5)%, 'edgecolor', [1, 0.41, 0.16])
        h_cdf = cdfplot(DeltaFR_of_TM); %, 'linewidth',1.5)
        set(h_cdf, 'linewidth', 2)
        hold on
    elseif modulated==0
        %         histogram(DeltaFR_of_NonM, 'binwidth', 0.2, 'DisplayStyle', 'stairs',...
        %             'linewidth',1.5)%, 'edgecolor', [0.5, 0.5, 0.5])
        h_cdf = cdfplot(DeltaFR_of_NonM);%, 'linewidth',1.5)
        set(h_cdf, 'linewidth', 2)
        hold on
    elseif modulated==2
        h_cdf = cdfplot([DeltaFR_of_TM; DeltaFR_of_NonM]);%, 'linewidth',1.5)
        set(h_cdf, 'linewidth', 2)
        hold on        
    end
    
    xlabel('\Delta FR (sp/sec)')
    ylabel('Cumulative Prob.')%('# of neurons')
    set(gca, 'fontsize', 14, 'box', 'off')
    %     hLegend = legend('Task-Modulated', 'Not Modulated', 'fontsize', 11);
    
    %     % title([BrainRegion, ', ', TM_test, ': ', num2str(sum(p_logic_neur)),...
    %     %     ' task-modulated neurons out of ', num2str(size(Neur_idx,1))],...
    %     %     'fontsize', 12, 'Interpreter','none')
    %     title({TM_test_title; [num2str(sum(p_logic_neur)),...
    %         ' task-modulated neurons out of ', num2str(size(Neur_idx,1))]},...
    %         'fontsize', 12, 'fontweight', 'normal', 'Interpreter','none')
    %     % title({TM_test; [num2str(sum(p_logic_neur)),...
    %     %     ' task-modulated neurons out of ', num2str(size(Neur_idx,1))]},...
    %     %     'fontsize', 12, 'fontweight', 'normal', 'Interpreter','none')
    %
    %     %percSignig = sum(p_logic_neur)/size(Neur_idx,1);
    
    totalTM(n) = sum(p_logic_neur);
    totalNeur(n) = size(Neur_idx,1);
    totalNeur_FanoFactor(n) = sum(~isnan(T.avg_ff_post_move(Neur_idx)));
    percTM(n) = totalTM(n)/totalNeur(n);
    disp([labs{n} , ': ', num2str(totalTM(n)),...
        ' task-modulated neurons out of ', num2str(totalNeur(n))])
    %disp([labs{n}, ' Fano Factor: ', num2str(totalNeur_FanoFactor)])
    
end

hLegend = legend(labs, 'fontsize', 11);
hold off

title({[TM_test_title, '; % task-modulated neurons per lab: '];...
    num2str(percTM*100)},...
    'fontsize', 13, 'fontweight', 'normal')%, 'Interpreter','none')

disp([labs; ' Fano Factor: ', num2str(totalNeur_FanoFactor)])

%--------------------------------------------------------------------------
% %To find a Task-Modulated neuron, for selecting/plotting an example neuron
% % from the brain region:
%
% DeltaFR_thresh = 5; %10; %select based on histogram plotted above
% N_NeurOverThresh = sum(DeltaFR_of_TM>DeltaFR_thresh);
% Neur_idx_TM = Neur_idx(p_logic_neur); %indices of all Task-Modulated neurons
% idx = Neur_idx_TM(DeltaFR_of_TM>DeltaFR_thresh); %indices of very (>Threshold) TM neurons
% Ex_clusterID = T.cluster_ids(idx);
% Ex_eid = T.eid(idx);
% Ex_pid = T.pid(idx);
% Ex_subj = T.subject(idx);
% T(idx,[2, 14:2:26]); %All TM test results for the particular neuron(s)
% %one where post stim is false: T(2813,14:2:26), DeltaFR_of_TM(Neur_idx_TM==2813)
% T(idx, [2,35:37]);


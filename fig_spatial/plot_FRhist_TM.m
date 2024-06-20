% Code for generating firing rate histograms of task-modulated neurons
% (e.g., in Fig. 7c-d, left side).
% Code written by: Marsa Taheri

function [hLegend] = plot_FRhist_TM(CSVfile, BrainRegion, TM_test)

T = readtable(CSVfile);

%Find neurons in specified brain region:
Neur_idx = find(strcmp(T.region, BrainRegion));
%For a specific lab:
%Neur_idx = find(strcmp(T.lab, 'churchlandlab_ucla') & strcmp(T.region, BrainRegion));
%Neur_idx = find(strcmp(T.lab, 'mainenlab') & strcmp(T.region, BrainRegion));


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


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
        TM_test_title = 'Stimulus vs. Pre-stim Test';
    case 'pre_move'
        avg_fr_pre_move_reg = T.avg_fr_pre_move(Neur_idx);
        FR0_TM = avg_fr_base_reg(p_logic_neur);
        FR1_TM = avg_fr_pre_move_reg(p_logic_neur);
        FR0_NonM = avg_fr_base_reg(p_logic_neur==0);
        FR1_NonM = avg_fr_pre_move_reg(p_logic_neur==0);
        TM_test_title = 'Movement Initiation vs. Pre-stim Test';
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
        TM_test_title = 'Movement vs. Pre-stim Test';
    case 'post_reward'
        avg_fr_post_reward_reg = T.avg_fr_post_reward(Neur_idx);
        FR0_TM = avg_fr_base_reg(p_logic_neur);
        FR1_TM = avg_fr_post_reward_reg(p_logic_neur);
        FR0_NonM = avg_fr_base_reg(p_logic_neur==0);
        FR1_NonM = avg_fr_post_reward_reg(p_logic_neur==0);
        TM_test_title = 'Reward vs. Pre-stim Test';
end

DeltaFR_of_TM = FR1_TM - FR0_TM;
DeltaFR_of_NonM = FR1_NonM - FR0_NonM;


histogram(DeltaFR_of_TM, 'binwidth', 0.2, 'DisplayStyle', 'stairs',...
    'linewidth',1.5, 'edgecolor', [1, 0.41, 0.16])
hold on
histogram(DeltaFR_of_NonM, 'binwidth', 0.2, 'DisplayStyle', 'stairs',...
    'linewidth',1.5, 'edgecolor', [0.5, 0.5, 0.5])

%xlabel('\Delta FR (pre-movement - pre-stim)')
ylabel('# of neurons')
set(gca, 'fontsize', 14, 'box', 'off')
hLegend = legend('Task-Modulated', 'Not Modulated', 'fontsize', 11);

title({TM_test_title; [num2str(sum(p_logic_neur)),...
    ' task-modulated neurons out of ', num2str(size(Neur_idx,1))]},...
    'fontsize', 12, 'fontweight', 'normal', 'Interpreter','none')
hold off

%--------------------------------------------------------------------------
%To find a Task-Modulated neuron, for selecting/plotting an example neuron 
% from the brain region:

DeltaFR_thresh = 5; %10; %select based on histogram plotted above
N_NeurOverThresh = sum(DeltaFR_of_TM>DeltaFR_thresh);
Neur_idx_TM = Neur_idx(p_logic_neur); %indices of all Task-Modulated neurons
idx = Neur_idx_TM(DeltaFR_of_TM>DeltaFR_thresh); %indices of very (>Threshold) TM neurons
Ex_clusterID = T.cluster_ids(idx);
Ex_eid = T.eid(idx);
Ex_pid = T.pid(idx);
Ex_subj = T.subject(idx);
T(idx,[2, 14:2:26]); %All TM test results for the particular neuron(s)
%one where post stim is false: T(2813,14:2:26), DeltaFR_of_TM(Neur_idx_TM==2813)
T(idx, [2,35:37]);


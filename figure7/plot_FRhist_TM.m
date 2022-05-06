% Code for generating firing rate histograms of task-modulated neurons
% (e.g., in Fig. 7c-d, left side).
% Code written by: Marsa Taheri

function [hLegend] = plot_FRhist_TM(CSVfile, BrainRegion, TM_test)

T = readtable(CSVfile);

%Find neurons in specified brain region:
Neur_idx = find(strcmp(T.region, BrainRegion));

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
    case 'pre_move_lr'
        avg_fr_pre_moveL_reg = T.avg_fr_pre_moveL(Neur_idx);
        avg_fr_pre_moveR_reg = T.avg_fr_pre_moveR(Neur_idx);
        FR0_TM = avg_fr_pre_moveL_reg(p_logic_neur);
        FR1_TM = avg_fr_pre_moveR_reg(p_logic_neur);
        FR0_NonM = avg_fr_pre_moveL_reg(p_logic_neur==0);
        FR1_NonM = avg_fr_pre_moveR_reg(p_logic_neur==0);
    case 'post_stim'
        avg_fr_post_stim_reg = T.avg_fr_post_stim(Neur_idx);
        FR0_TM = avg_fr_base_reg(p_logic_neur);
        FR1_TM = avg_fr_post_stim_reg(p_logic_neur);
        FR0_NonM = avg_fr_base_reg(p_logic_neur==0);
        FR1_NonM = avg_fr_post_stim_reg(p_logic_neur==0);
end

DeltaFR_of_TM = FR1_TM - FR0_TM;
DeltaFR_of_NonM = FR1_NonM - FR0_NonM;


histogram(DeltaFR_of_NonM, 'binwidth', 0.2, 'DisplayStyle', 'stairs',...
    'linewidth',1.5, 'edgecolor', [0.5, 0.5, 0.5])
hold on
histogram(DeltaFR_of_TM, 'binwidth', 0.2, 'DisplayStyle', 'stairs',...
    'linewidth',1.5, 'edgecolor', [1, 0.41, 0.16])
%xlabel('\Delta FR (pre-movement - pre-stim)')
ylabel('# of neurons')
set(gca, 'fontsize', 14, 'box', 'off')
hLegend = legend('Not Modulated', 'Task-Modulated', 'fontsize', 11);

% title([BrainRegion, ', ', TM_test, ': ', num2str(sum(p_logic_neur)),...
%     ' task-modulated neurons out of ', num2str(size(Neur_idx,1))],...
%     'fontsize', 12, 'Interpreter','none')
title({TM_test; [num2str(sum(p_logic_neur)),...
    ' task-modulated neurons out of ', num2str(size(Neur_idx,1))]},...
    'fontsize', 12, 'fontweight', 'normal', 'Interpreter','none')

%percSignig = sum(p_logic_neur)/size(Neur_idx,1);



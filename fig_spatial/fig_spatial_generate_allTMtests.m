function [] = fig_spatial_generate_allTMtests(CSVfile, BrainRegion, save_path)

TM_test1 = 'start_to_move'; %'post_stim'; %'trial'; %'post_reward';%
TM_test2 = 'pre_move_lr'; %'pre_move'; %'post_move'; %
TM_test3 = 'post_stim'; %'trial'; %'post_reward';%
TM_test4 = 'pre_move'; %'post_move'; %
TM_test5 = 'post_move'; %
TM_test6 = 'post_reward';%
%TM_test7 = 'trial'; %'post_reward';% Old test, no longer used


f = figure;
ax(1) = subplot(3,2,1);
[hLegend1, pMWU_corrTM1, mdlTM1, mdlTM1_shuffle] = plot3D_TM(CSVfile,...
    BrainRegion, TM_test1, [0,0,0], ax(1), save_path);
%set(hLegend1, 'box', 'on', 'Position', [0.7310  0.4277  0.1713  0.0363])
%set(get(ax(2), 'Ylabel'), 'Rotation', rot, 'Position', pos2)

figure(f)
ax(2) = subplot(3,2,2);
[hLegend2, pMWU_corrTM2, mdlTM2, mdlTM2_shuffle] = plot3D_TM(CSVfile,...
    BrainRegion, TM_test2, [0,0,0], ax(2), save_path);
%set(hLegend2, 'box', 'on', 'Position', [0.7445  0.1285  0.1619 0.0342])
%set(get(ax(3), 'Ylabel'), 'Rotation', rot, 'Position', pos3)

figure(f)
ax(3) = subplot(3,2,3);
[~, pMWU_corrTM3] = plot3D_TM(CSVfile,...
    BrainRegion, TM_test3, [0,0,0], ax(3), save_path);

figure(f)
ax(4) = subplot(3,2,4);
[~, pMWU_corrTM4] = plot3D_TM(CSVfile,...
    BrainRegion, TM_test4, [0,0,0], ax(4), save_path);

figure(f)
ax(5) = subplot(3,2,5);
[~, pMWU_corrTM5] = plot3D_TM(CSVfile,...
    BrainRegion, TM_test5, [0,0,0], ax(5), save_path);

figure(f)
ax(6) = subplot(3,2,6);
[~, pMWU_corrTM6] = plot3D_TM(CSVfile,...
    BrainRegion, TM_test6, [0,0,0], ax(6), save_path);


%figure(f)
%f.Position = [996 38 797 979];

%% Display signif. difference in x, y, z, WF amp, and WF p2t duration:
disp({'x','y','z','amp','p2t dur'})
disp([pMWU_corrTM1; pMWU_corrTM2; pMWU_corrTM3; pMWU_corrTM4;...
    pMWU_corrTM5; pMWU_corrTM6])

disp({'x','y','z','amp','p2t dur'})
disp([pMWU_corrTM1; pMWU_corrTM2; pMWU_corrTM3; pMWU_corrTM4;...
    pMWU_corrTM5; pMWU_corrTM6]<0.05)

%saveas(f, append(save_path, 'figure_spatial_', BrainRegion, '.png'))

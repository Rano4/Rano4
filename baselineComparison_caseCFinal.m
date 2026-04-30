%% baselineComparison_caseCFinal.m
% Compares ANN detector against:
%   (a) Logistic Regression (linear baseline)
%   (b) Random Forest (ensemble baseline)
%   (c) Hard threshold on dObs alone
%
% Uses same time-based 60/20/20 split as trainANN_caseCFinal.m
% Produces unified ROC/PR comparison figure for journal paper.

clearvars; clc; close all; rng(0);

%% ---- Load dataset -----------------------------------------------------
outDir = fullfile(fileparts(mfilename('fullpath')), 'caseCFinal');
load(fullfile(outDir, 'isl_dataset_caseCFinal.mat'));
load(fullfile(outDir, 'islAnnModel_caseCFinal.mat'), 'bestNet','bestThr','mu_tr','std_tr');

N = size(X_data, 1);

%% ---- Same time-based split as training --------------------------------
nPerSlot = numSat * cfg.Kcand;
trainEnd = round(0.60 * Tuse) * nPerSlot;
valEnd   = round(0.80 * Tuse) * nPerSlot;

Xtr = X_data(1:trainEnd,:);          Ytr = Y_data(1:trainEnd);
Xva = X_data(trainEnd+1:valEnd,:);   Yva = Y_data(trainEnd+1:valEnd);
Xte = X_data(valEnd+1:end,:);        Yte = Y_data(valEnd+1:end);

fprintf('Test set: %d samples, fail rate=%.4f\n', numel(Yte), mean(Yte));

%% ---- Fit baseline models ----------------------------------------------

% (a) Logistic Regression
fprintf('Fitting Logistic Regression...\n');
Mdl_LR = fitclinear([Xtr; Xva], [Ytr; Yva], ...
    'Learner','logistic', 'Lambda',1e-3, 'ClassNames',[0;1]);

% (b) Random Forest (100 trees, class-weighted)
fprintf('Fitting Random Forest...\n');
n0 = sum([Ytr;Yva]==0); n1 = sum([Ytr;Yva]==1);
cost_rf = [0 1; max(1,n0/n1) 0];
Mdl_RF = fitcensemble([Xtr; Xva], [Ytr; Yva], ...
    'Method','Bag', 'NumLearningCycles',100, ...
    'ClassNames',[0;1], 'Cost',cost_rf, ...
    'Learners', templateTree('MaxNumSplits',20));

%% ---- ANN scores on test set -------------------------------------------
Xte_n  = (Xte - mu_tr) ./ (std_tr + 1e-8);
pANN   = bestNet(Xte_n')';

%% ---- Baseline scores --------------------------------------------------
[~, sc_LR] = predict(Mdl_LR, Xte);
[~, sc_RF] = predict(Mdl_RF, Xte);

cls_LR = Mdl_LR.ClassNames;
cls_RF = Mdl_RF.ClassNames;
idx1_LR = find(double(cls_LR(:))==1,1);
idx1_RF = find(double(cls_RF(:))==1,1);
pLR = sc_LR(:, idx1_LR);
pRF = sc_RF(:, idx1_RF);

% Hard threshold on dObs
dObs_te = Xte(:,1);

%% ---- Metrics ----------------------------------------------------------
Yte_d = double(Yte);

[roc_xA, roc_yA, ~, AUC_A] = perfcurve(Yte_d, pANN, 1);
[roc_xL, roc_yL, ~, AUC_L] = perfcurve(Yte_d, pLR,  1);
[roc_xR, roc_yR, ~, AUC_R] = perfcurve(Yte_d, pRF,  1);
[roc_xD, roc_yD, ~, AUC_D] = perfcurve(Yte_d, dObs_te, 1);

[pr_xA, pr_yA, ~, AUPR_A] = perfcurve(Yte_d, pANN,    1,'xCrit','reca','yCrit','prec');
[pr_xL, pr_yL, ~, AUPR_L] = perfcurve(Yte_d, pLR,     1,'xCrit','reca','yCrit','prec');
[pr_xR, pr_yR, ~, AUPR_R] = perfcurve(Yte_d, pRF,     1,'xCrit','reca','yCrit','prec');
[pr_xD, pr_yD, ~, AUPR_D] = perfcurve(Yte_d, dObs_te, 1,'xCrit','reca','yCrit','prec');

% Threshold-optimized metrics for each model
models_pFail = {pANN, pLR, pRF};
models_names = {'ANN', 'LogReg', 'RandForest'};
fprintf('\n%-14s  AUC     AUPR    F1      Acc     MCC\n','Method');
for m = 1:3
    p_m = models_pFail{m};
    % Threshold at Youden index
    [fpr_g, tpr_g, thr_g] = perfcurve(Yte_d, p_m, 1);
    [~, ibest] = max(tpr_g - fpr_g);
    thr_best = thr_g(ibest);
    yhat_m = double(p_m >= thr_best);

    tp = sum(yhat_m & Yte_d);  fp = sum(yhat_m & ~Yte_d);
    tn = sum(~yhat_m & ~Yte_d); fn = sum(~yhat_m & Yte_d);
    acc_m = (tp+tn)/(tp+fp+tn+fn);
    pre_m = tp/(tp+fp+1e-9);
    rec_m = tp/(tp+fn+1e-9);
    f1_m  = 2*pre_m*rec_m/(pre_m+rec_m+1e-9);
    mcc_m = (tp*tn-fp*fn)/sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)+1e-9);

    switch m
        case 1
            aupr_m = AUPR_A; auc_m = AUC_A;
        case 2
            aupr_m = AUPR_L; auc_m = AUC_L;
        case 3
            aupr_m = AUPR_R; auc_m = AUC_R;
    end
    fprintf('%-14s  %.4f  %.4f  %.4f  %.4f  %.4f\n', ...
        models_names{m}, auc_m, aupr_m, f1_m, acc_m, mcc_m);
end

% Hard threshold dObs>=3000
yhat_ht = double(dObs_te >= 3000);
tp=sum(yhat_ht & Yte_d); fp=sum(yhat_ht & ~Yte_d);
tn=sum(~yhat_ht & ~Yte_d); fn=sum(~yhat_ht & Yte_d);
f1_ht = 2*(tp/(tp+fp+1e-9))*(tp/(tp+fn+1e-9)) / ...
    ((tp/(tp+fp+1e-9))+(tp/(tp+fn+1e-9))+1e-9);
acc_ht=(tp+tn)/(tp+fp+tn+fn);
mcc_ht=(tp*tn-fp*fn)/sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)+1e-9);
fprintf('%-14s  %.4f  %.4f  %.4f  %.4f  %.4f\n', ...
    'Threshold(3000)', AUC_D, AUPR_D, f1_ht, acc_ht, mcc_ht);

%% ---- Figure: ROC comparison ------------------------------------------
fig1 = figure('Color','w','Position',[100 100 680 500]);
hold on; box on; grid on;

plot(roc_xA, roc_yA, 'b-',  'LineWidth',2.2, 'DisplayName', sprintf('ANN          (AUC=%.4f)', AUC_A));
plot(roc_xR, roc_yR, 'g--', 'LineWidth',2.0, 'DisplayName', sprintf('Random Forest (AUC=%.4f)', AUC_R));
plot(roc_xL, roc_yL, 'm:',  'LineWidth',2.0, 'DisplayName', sprintf('Logistic Reg. (AUC=%.4f)', AUC_L));
plot(roc_xD, roc_yD, 'r-.', 'LineWidth',1.8, 'DisplayName', sprintf('dObs Threshold (AUC=%.4f)', AUC_D));
plot([0 1],[0 1], 'k--', 'LineWidth',1.0, 'DisplayName','Random');

xlabel('False Positive Rate', 'FontSize',13);
ylabel('True Positive Rate',  'FontSize',13);
title('ROC Curves — Detector Comparison', 'FontSize',13,'FontWeight','bold');
legend('Location','southeast','FontSize',10);
set(gca,'FontSize',12);

saveas(fig1, fullfile(outDir, 'fig_roc_comparison_caseCFinal.png'));
exportgraphics(fig1, fullfile(outDir, 'fig_roc_comparison_caseCFinal.pdf'), 'ContentType','vector');

%% ---- Figure: PR comparison -------------------------------------------
fig2 = figure('Color','w','Position',[100 100 680 500]);
hold on; box on; grid on;

plot(pr_xA, pr_yA, 'b-',  'LineWidth',2.2, 'DisplayName', sprintf('ANN          (AUPR=%.4f)', AUPR_A));
plot(pr_xR, pr_yR, 'g--', 'LineWidth',2.0, 'DisplayName', sprintf('Random Forest (AUPR=%.4f)', AUPR_R));
plot(pr_xL, pr_yL, 'm:',  'LineWidth',2.0, 'DisplayName', sprintf('Logistic Reg. (AUPR=%.4f)', AUPR_L));
plot(pr_xD, pr_yD, 'r-.', 'LineWidth',1.8, 'DisplayName', sprintf('dObs Threshold (AUPR=%.4f)', AUPR_D));
yline(mean(Yte_d), 'k--', 'LineWidth',1.0, 'DisplayName', sprintf('Chance (%.4f)', mean(Yte_d)));

xlabel('Recall',    'FontSize',13);
ylabel('Precision', 'FontSize',13);
title('PR Curves — Detector Comparison', 'FontSize',13,'FontWeight','bold');
legend('Location','northeast','FontSize',10);
set(gca,'FontSize',12);

saveas(fig2, fullfile(outDir, 'fig_pr_comparison_caseCFinal.png'));
exportgraphics(fig2, fullfile(outDir, 'fig_pr_comparison_caseCFinal.pdf'), 'ContentType','vector');

fprintf('\nComparison figures saved.\n');

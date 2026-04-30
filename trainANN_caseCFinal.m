%% trainANN_caseCFinal.m
% Trains ANN-based ISL failure detector (binary classification).
% Time-based 60/20/20 train/val/test split to prevent leakage.
% Grid search: 7 architectures x 2 activations x 4 lambda values = 56 configs.
% Best model selected by AUPR on validation set.

clearvars; clc; rng(0);

%% ---- Load dataset -----------------------------------------------------
outDir = fullfile(fileparts(mfilename('fullpath')), 'caseCFinal');
load(fullfile(outDir, 'isl_dataset_caseCFinal.mat'));

N = size(X_data, 1);
fprintf('Loaded dataset: N=%d, features=%d, fail_rate=%.4f\n', ...
    N, size(X_data,2), mean(Y_data));

%% ---- Time-based split -------------------------------------------------
% Samples are ordered [t=1..Tuse, sat=1..numSat, k=1..Kcand]
nPerSlot  = numSat * cfg.Kcand;
trainEnd  = round(0.60 * Tuse) * nPerSlot;
valEnd    = round(0.80 * Tuse) * nPerSlot;

idx_train = 1:trainEnd;
idx_val   = trainEnd+1:valEnd;
idx_test  = valEnd+1:N;

Xtr = X_data(idx_train,:);  Ytr = Y_data(idx_train);
Xva = X_data(idx_val,:);    Yva = Y_data(idx_val);
Xte = X_data(idx_test,:);   Yte = Y_data(idx_test);

fprintf('Train=%d  Val=%d  Test=%d\n', numel(idx_train), numel(idx_val), numel(idx_test));

%% ---- Normalise (z-score on train statistics) -------------------------
mu_tr  = mean(Xtr);
std_tr = std(Xtr) + 1e-8;
Xtr_n  = (Xtr - mu_tr) ./ std_tr;
Xva_n  = (Xva - mu_tr) ./ std_tr;
Xte_n  = (Xte - mu_tr) ./ std_tr;

%% ---- Class weights (inverse-frequency) --------------------------------
w_pos  = sum(~Ytr) / sum(Ytr);
fprintf('Class weight w_pos=%.3f\n', w_pos);

%% ---- Hyperparameter grid ----------------------------------------------
hiddenSizes  = {[64], [128], [64 32], [128 64], [256 128], [128 64 32], [256 128 64]};
activations  = {'relu', 'tanh'};
lambdaVals   = [0, 1e-4, 1e-3, 1e-2];

nConfigs = numel(hiddenSizes) * numel(activations) * numel(lambdaVals);
results  = struct('arch',{}, 'act',{}, 'lam',{}, 'aupr_val',{}, 'auc_val',{});

bestAUPR = -Inf;
bestNet  = [];
bestCfg  = [];

fprintf('\nRunning %d configurations...\n', nConfigs);
cfgCount = 0;

for ha = 1:numel(hiddenSizes)
    for ac = 1:numel(activations)
        for la = 1:numel(lambdaVals)
            cfgCount = cfgCount + 1;
            hs  = hiddenSizes{ha};
            act = activations{ac};
            lam = lambdaVals(la);

            %% Build network
            net = patternnet(hs, 'trainscg');
            net.trainParam.showWindow   = false;
            net.trainParam.epochs       = 300;
            net.trainParam.max_fail     = 15;
            net.trainParam.lr           = 0.005;
            net.performParam.regularization = lam;

            for L = 1:numel(hs)
                net.layers{L}.transferFcn = act;
            end

            % Validation split from training data
            net.divideFcn  = 'divideind';
            nTr2 = round(0.85 * numel(idx_train));
            net.divideParam.trainInd = 1:nTr2;
            net.divideParam.valInd   = nTr2+1:numel(idx_train);
            net.divideParam.testInd  = [];

            % Class weights via sample weights
            sampleWeights = double(~Ytr) + w_pos * double(Ytr);

            [net, ~] = train(net, Xtr_n', double(Ytr)', 'SampleWeights', sampleWeights');

            %% Evaluate on validation set
            yhat_va = net(Xva_n')';
            [~,~,~, auc_va]  = perfcurve(double(Yva), yhat_va, 1);
            [pr_x, pr_y, ~]  = perfcurve(double(Yva), yhat_va, 1, 'xCrit','reca','yCrit','prec');
            aupr_va = trapz(pr_x, pr_y);

            results(cfgCount).arch     = hs;
            results(cfgCount).act      = act;
            results(cfgCount).lam      = lam;
            results(cfgCount).aupr_val = aupr_va;
            results(cfgCount).auc_val  = auc_va;

            archStr = mat2str(hs);
            fprintf('[%2d/%d] arch=%-16s act=%-4s lam=%.0e  AUPR=%.4f  AUC=%.4f\n', ...
                cfgCount, nConfigs, archStr, act, lam, aupr_va, auc_va);

            if aupr_va > bestAUPR
                bestAUPR = aupr_va;
                bestNet  = net;
                bestCfg  = struct('arch',{hs},'act',act,'lam',lam);
            end
        end
    end
end

fprintf('\nBest val AUPR=%.4f  arch=%s  act=%s  lam=%.0e\n', ...
    bestAUPR, mat2str(bestCfg.arch), bestCfg.act, bestCfg.lam);

%% ---- Threshold optimisation on validation set -------------------------
yhat_va = bestNet(Xva_n')';
thresholds = 0.05:0.01:0.95;
bestF1 = -Inf; bestThr = 0.5;
for thr = thresholds
    pred = yhat_va >= thr;
    tp = sum(pred & Yva);
    fp = sum(pred & ~Yva);
    fn = sum(~pred & Yva);
    f1 = 2*tp / (2*tp + fp + fn + 1e-9);
    if f1 > bestF1, bestF1 = f1; bestThr = thr; end
end
fprintf('Optimal threshold (val F1): thr=%.2f  F1=%.4f\n', bestThr, bestF1);

%% ---- Final evaluation on test set ------------------------------------
yhat_te = bestNet(Xte_n')';
ypred_te = yhat_te >= bestThr;

TP = sum( ypred_te &  Yte);
FP = sum( ypred_te & ~Yte);
TN = sum(~ypred_te & ~Yte);
FN = sum(~ypred_te &  Yte);

precision  = TP / (TP + FP + 1e-9);
recall     = TP / (TP + FN + 1e-9);
f1_te      = 2*precision*recall / (precision + recall + 1e-9);
acc_te     = (TP + TN) / (TP + FP + TN + FN);
mcc_te     = (TP*TN - FP*FN) / sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN) + 1e-9);

[~,~,~,auc_te] = perfcurve(double(Yte), yhat_te, 1);
[pr_x_te, pr_y_te] = perfcurve(double(Yte), yhat_te, 1, 'xCrit','reca','yCrit','prec');
aupr_te = trapz(pr_x_te, pr_y_te);

fprintf('\n===== TEST SET RESULTS =====\n');
fprintf('AUC   = %.4f\n', auc_te);
fprintf('AUPR  = %.4f\n', aupr_te);
fprintf('F1    = %.4f\n', f1_te);
fprintf('Acc   = %.4f\n', acc_te);
fprintf('MCC   = %.4f\n', mcc_te);
fprintf('Prec  = %.4f\n', precision);
fprintf('Rec   = %.4f\n', recall);
fprintf('Confusion: TN=%d FP=%d FN=%d TP=%d\n', TN, FP, FN, TP);

%% ---- Permutation feature importance ----------------------------------
fprintf('\nPermutation feature importance (AUC drop):\n');
baseAUC = auc_te;
for f = 1:numel(featureNames)
    Xte_perm = Xte_n;
    Xte_perm(:,f) = Xte_perm(randperm(size(Xte_perm,1)), f);
    yhat_p = bestNet(Xte_perm')';
    [~,~,~,auc_p] = perfcurve(double(Yte), yhat_p, 1);
    fprintf('  %-10s : drop = %.4f\n', featureNames{f}, baseAUC - auc_p);
end

%% ---- ROC / PR figures -------------------------------------------------
figure('Name','ROC Curve','Color','w');
[roc_x, roc_y] = perfcurve(double(Yte), yhat_te, 1);
plot(roc_x, roc_y, 'b-', 'LineWidth', 1.8); hold on;
plot([0 1],[0 1],'k--');
xlabel('False Positive Rate'); ylabel('True Positive Rate');
title(sprintf('ROC Curve — AUC = %.4f', auc_te));
grid on; legend(sprintf('ANN (AUC=%.4f)', auc_te), 'Random', 'Location','se');
saveas(gcf, fullfile(outDir, 'roc_caseCFinal.png'));

figure('Name','PR Curve','Color','w');
plot(pr_x_te, pr_y_te, 'r-', 'LineWidth', 1.8);
xlabel('Recall'); ylabel('Precision');
title(sprintf('PR Curve — AUPR = %.4f', aupr_te));
grid on;
saveas(gcf, fullfile(outDir, 'pr_caseCFinal.png'));

%% ---- Save model -------------------------------------------------------
save(fullfile(outDir, 'islAnnModel_caseCFinal.mat'), ...
    'bestNet', 'bestCfg', 'bestThr', 'mu_tr', 'std_tr', ...
    'auc_te', 'aupr_te', 'f1_te', 'acc_te', 'mcc_te', ...
    'featureNames', 'results');

fprintf('\nModel saved to islAnnModel_caseCFinal.mat\n');

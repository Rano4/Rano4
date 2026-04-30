%% buildDataset_caseCFinal.m
% Builds synthetic ISL failure detection dataset from dati2 geometry.
% Labels use additive risk score: distance margin + clearance margin + trend + stochastic.

clearvars; clc; rng(42);

%% ---- Load geometry ----------------------------------------------------
outDir = fullfile(fileparts(mfilename('fullpath')), 'caseCFinal');
load(fullfile(outDir, 'dati2_caseCFinal.mat'));

%% ---- Dataset config ---------------------------------------------------
cfg.RtKm        = 6371;
cfg.TmaxCap     = 2000;   % max time slots to use
cfg.H           = 60;
cfg.Kcand       = 8;
cfg.DmaxNow     = 4500;
cfg.DmaxFut     = 3600;
cfg.clrRiskRef  = 500;    % km — clearance reference for risk score
cfg.wRange      = 0.70;   % weight: distance margin risk
cfg.wBlock      = 0.55;   % weight: clearance risk
cfg.wTrend      = 0.10;   % weight: distance trend
cfg.latentSigma = 1.50;   % unmodeled risk noise
cfg.pStochFail  = 0.06;   % baseline stochastic failure probability

% Observation noise
cfg.sigD    = 600;         % km  noise on distance measurement
cfg.sigDD   = 25.0;        % km/slot noise on distance rate
cfg.sigClr  = 150;         % km  noise on clearance margin
cfg.sigRat  = 0.05;        % noise on clearance/distance ratio
cfg.sigNorm = 0.05;        % noise on normalized features

Tuse = min(cfg.TmaxCap, T);

%% ---- Build dataset ----------------------------------------------------
fprintf('Building ISL failure dataset (T=%d, numSat=%d, K=%d)...\n', Tuse, numSat, cfg.Kcand);

nRows  = Tuse * numSat * cfg.Kcand;
X_data = zeros(nRows, 5);  % [dObs, clrObs, ddObs, ratioObs, normObs]
Y_data = false(nRows, 1);
row    = 0;

for t = 1:Tuse
    for i = 1:numSat
        for k = 1:cfg.Kcand
            row = row + 1;

            dTrue   = distNow(i,k,t);
            dFTrue  = distFut(i,k,t);
            ddTrue  = ddNow(i,k,t);
            clrTrue = clrFut(i,k,t);

            % Noisy observations
            dObs   = max(100, dTrue   + cfg.sigD   * randn);
            ddObs  = ddTrue  + cfg.sigDD  * randn;
            clrObs = clrTrue + cfg.sigClr * randn;

            % clrOverD — clamped to avoid divide-by-near-zero
            ratioObs = max(0, min(2.0, clrObs / max(dObs, 100) + cfg.sigRat * randn));

            % Normalised distance (0=close, 1=at DmaxNow)
            normObs  = max(0, min(1, dObs / cfg.DmaxNow + cfg.sigNorm * randn));

            X_data(row,:) = [dObs, clrObs, ddObs, ratioObs, normObs];

            % Label
            Y_data(row) = sampleGeomLabel(ddTrue, dFTrue, clrTrue, cfg);
        end
    end
end
X_data = X_data(1:row,:);
Y_data = Y_data(1:row);

%% ---- Statistics -------------------------------------------------------
nFail = sum(Y_data);
nOk   = sum(~Y_data);
fprintf('\nDataset: %d samples,  Fail=%d (%.4f),  OK=%d\n', ...
    row, nFail, nFail/row, nOk);

detFail = sum(distFut(:) <= 0);
fprintf('Deterministic failures (dFut<=0): %d\n', detFail);

featureNames = {'dObs','clrObs','ddObs','ratioObs','normObs'};
for f = 1:5
    fprintf('  %8s: mean=%8.2f  std=%7.2f\n', ...
        featureNames{f}, mean(X_data(:,f)), std(X_data(:,f)));
end

%% ---- Save -------------------------------------------------------------
save(fullfile(outDir, 'isl_dataset_caseCFinal.mat'), ...
    'X_data', 'Y_data', 'featureNames', 'cfg', 'Tuse', 'numSat');
fprintf('\nisl_dataset_caseCFinal.mat saved.\n');

%% ---- Label function ---------------------------------------------------
function label = sampleGeomLabel(ddNow_val, dFut, clrFut, cfg)
    if clrFut <= 0
        label = true; return;
    end
    marginRange = (dFut - cfg.DmaxFut) / 300;
    marginBlock = (cfg.clrRiskRef - clrFut) / 200;
    trendRisk   = ddNow_val / 30;
    latent      = cfg.latentSigma * randn;
    riskScore   = cfg.wRange * marginRange + cfg.wBlock * marginBlock ...
                + cfg.wTrend * trendRisk   + latent;
    pGeom  = 1 / (1 + exp(-riskScore));
    pFail  = 1 - (1 - pGeom) * (1 - cfg.pStochFail);
    label  = (rand < pFail);
end

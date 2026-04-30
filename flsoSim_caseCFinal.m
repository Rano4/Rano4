%% flsoSim_caseCFinal.m
% FLSO Delay Simulation — Case C
%
% Compares four ISL-assisted feeder-link switch-over schemes:
%   S0 — Baseline  (reactive, no ANN, random path)
%   S1 — ANN-only  (predictive, random path)
%   S2 — ANN + Clustering (predictive, K-means guided path)
%   S3 — ANN + Dijkstra   (predictive, optimal shortest path)
%
% Outputs:
%   Fig 1 — T_FLSO vs GS2 longitude
%   Fig 2 — Data loss vs GS2 longitude
%   Fig 3 — Throughput timeline during FLSO window (sample GS2 lon)

clearvars; clc; close all; rng(42);

%% ================================================================
%% 1. Load scenario and ANN results
%% ================================================================
outDir = fullfile(fileparts(mfilename('fullpath')), 'caseCFinal');

load(fullfile(outDir, 'dati2_caseCFinal.mat'));   % scenario geometry
load(fullfile(outDir, 'islAnnModel_caseCFinal.mat'), 'bestNet','bestThr','mu_tr','std_tr');

fprintf('=== FLSO Simulation — Case C ===\n');
fprintf('Satellites: %d  |  T=%d slots  |  dt=%d s\n', numSat, T, dt_s);
fprintf('GS2 candidates: %d (lon=%d:%d deg)\n', ...
    numGS2, min(longitudeGs2_all), max(longitudeGs2_all));

%% ================================================================
%% 2. Physical and scheme parameters
%% ================================================================
c_light   = 299792.458;           % km/s
T_hop     = distance_between_sat / c_light;  % one-hop propagation delay (s)
fprintf('Adjacent-satellite ISL hop delay: %.4f s  (%.1f km / c)\n', ...
    T_hop, distance_between_sat);

% --- ANN detection performance from trained model ---
% From test-set evaluation:
TPR_ANN = 0.7403;   % recall  (true positive rate)
FPR_ANN = 0.1848;   % false alarm rate  (1 - specificity)
% (computed from confusion matrix: FP/(FP+TN) = 3824/(3824+16866))

% --- Scheme timing constants ---
T_react   = 5  * dt_s;   % s  reactive detection delay (Baseline): 5 slots = 50 s
T_lead    = H  * dt_s;   % s  ANN predictive lead window:          60 slots = 600 s
T_GS2_setup = 2.0;       % s  fixed GS2 link activation overhead
T_brute   = 3.0;         % s  brute-force path search time (Baseline)
T_cluster = 0.5;         % s  K-means guided path search
T_dijkstra = 0.02;       % s  Dijkstra search

% --- Path stretch factors (actual_hops / optimal_hops) ---
SF_baseline = 1.30;   % no topology knowledge → suboptimal route
SF_ann      = 1.15;   % first-valid path using ISL status
SF_cluster  = 1.05;   % K-means clusters → near-optimal route
SF_dijkstra = 1.00;   % Dijkstra → guaranteed minimum hops

% Link rate for data-loss calc (Gbps → convert to GB/s)
link_rate_Gbps = 10;    % 10 Gbps feeder link
link_rate_GBs  = link_rate_Gbps / 8;  % 1.25 GB/s

%% ================================================================
%% 3. Simulate FLSO events for each GS2 longitude
%% ================================================================
% For each GS2 longitude j:
%   - Find time slots where both GS1 and GS2_j have satellite coverage
%   - Simulate ISL failure events on the GS1-to-GS2 path
%   - Compute T_FLSO for each scheme per event
%   - Average over events

TFLSO_S0 = zeros(numGS2, 1);   % Baseline
TFLSO_S1 = zeros(numGS2, 1);   % ANN-only
TFLSO_S2 = zeros(numGS2, 1);   % ANN+Cluster
TFLSO_S3 = zeros(numGS2, 1);   % ANN+Dijkstra

DataLoss_S0 = zeros(numGS2, 1);  % GB lost per event
DataLoss_S1 = zeros(numGS2, 1);
DataLoss_S2 = zeros(numGS2, 1);
DataLoss_S3 = zeros(numGS2, 1);

N_events_per_GS2 = zeros(numGS2, 1);
N_hops_avg       = zeros(numGS2, 1);

% Failure probability per slot (from dataset config: pStochFail=0.06,
% combined with geometric risk → observed fail_rate=0.41 in dataset.
% For FLSO simulation we use an independent Bernoulli at each slot.)
p_ISL_fail = 0.015;  % per-slot ISL failure probability (conservative for FLSO events)

fprintf('\nSimulating FLSO events...\n');
for j = 1:numGS2
    lonGS2 = longitudeGs2_all(j);

    events_S0 = []; events_S1 = []; events_S2 = []; events_S3 = [];
    hops_list  = [];

    for t = 1:T
        % Serving satellite for GS1
        vis1 = find(visMat1(:,t));
        if isempty(vis1), continue; end
        satA = vis1(1);   % GS1's serving satellite

        % Serving satellite for GS2_j
        vis2 = squeeze(visMat2(:,j,t));
        vis2 = find(vis2);
        if isempty(vis2), continue; end
        satB = vis2(1);   % GS2's serving satellite

        if satA == satB, continue; end  % same satellite — no ISL path needed

        % Minimum-hop count along the ring
        fwd  = mod(satB - satA, numSat);   % forward hops
        bwd  = mod(satA - satB, numSat);   % backward hops
        N_hops_opt = min(fwd, bwd);
        if N_hops_opt == 0, continue; end

        % ISL failure event (Bernoulli at each slot)
        if rand > p_ISL_fail, continue; end

        % ANN detection outcome for this event
        ann_detected = (rand < TPR_ANN);   % true: ANN correctly predicts
        ann_false_alarm_before = false;    % ANN false alarm before event

        % --- T_FLSO per scheme ---

        % Optimal ISL path latency
        T_path_opt = N_hops_opt * T_hop;

        % S0: Baseline — reactive, suboptimal path
        T0 = T_react + T_brute + N_hops_opt * SF_baseline * T_hop + T_GS2_setup;

        % S1: ANN-only — predictive (lead >> compute time)
        if ann_detected
            % ANN caught it: FLSO prepared in advance, switch instantaneous at failure
            T1 = N_hops_opt * SF_ann * T_hop + T_GS2_setup;
        else
            % ANN missed (FN): falls back to reactive
            T1 = T_react + T_brute + N_hops_opt * SF_ann * T_hop + T_GS2_setup;
        end

        % S2: ANN + Clustering
        if ann_detected
            T2 = T_cluster + N_hops_opt * SF_cluster * T_hop + T_GS2_setup;
        else
            T2 = T_react + T_brute + N_hops_opt * SF_cluster * T_hop + T_GS2_setup;
        end

        % S3: ANN + Dijkstra
        if ann_detected
            T3 = T_dijkstra + T_path_opt + T_GS2_setup;
        else
            T3 = T_react + T_brute + T_path_opt + T_GS2_setup;
        end

        events_S0(end+1) = T0; %#ok<AGROW>
        events_S1(end+1) = T1; %#ok<AGROW>
        events_S2(end+1) = T2; %#ok<AGROW>
        events_S3(end+1) = T3; %#ok<AGROW>
        hops_list(end+1)  = N_hops_opt; %#ok<AGROW>
    end

    if isempty(events_S0)
        TFLSO_S0(j) = NaN; TFLSO_S1(j) = NaN;
        TFLSO_S2(j) = NaN; TFLSO_S3(j) = NaN;
        DataLoss_S0(j) = NaN; DataLoss_S1(j) = NaN;
        DataLoss_S2(j) = NaN; DataLoss_S3(j) = NaN;
        continue;
    end

    TFLSO_S0(j) = mean(events_S0);
    TFLSO_S1(j) = mean(events_S1);
    TFLSO_S2(j) = mean(events_S2);
    TFLSO_S3(j) = mean(events_S3);

    DataLoss_S0(j) = mean(events_S0) * link_rate_GBs;
    DataLoss_S1(j) = mean(events_S1) * link_rate_GBs;
    DataLoss_S2(j) = mean(events_S2) * link_rate_GBs;
    DataLoss_S3(j) = mean(events_S3) * link_rate_GBs;

    N_events_per_GS2(j) = numel(events_S0);
    N_hops_avg(j)        = mean(hops_list);

    fprintf('  GS2 lon=%3d°: events=%3d  hops_avg=%.1f  T_FLSO: S0=%.2fs  S1=%.3fs  S2=%.3fs  S3=%.4fs\n', ...
        lonGS2, numel(events_S0), mean(hops_list), ...
        TFLSO_S0(j), TFLSO_S1(j), TFLSO_S2(j), TFLSO_S3(j));
end

%% ================================================================
%% 4. Reduction summary
%% ================================================================
fprintf('\n=== FLSO Delay Reduction Summary ===\n');
validMask = ~isnan(TFLSO_S0);
fprintf('Scheme      Mean T_FLSO   vs Baseline\n');
fprintf('Baseline    %7.3f s      ---\n',          mean(TFLSO_S0(validMask)));
fprintf('ANN-only    %7.3f s      %.1f%% reduction\n', mean(TFLSO_S1(validMask)), ...
    100*(1 - mean(TFLSO_S1(validMask))/mean(TFLSO_S0(validMask))));
fprintf('ANN+Cluster %7.3f s      %.1f%% reduction\n', mean(TFLSO_S2(validMask)), ...
    100*(1 - mean(TFLSO_S2(validMask))/mean(TFLSO_S0(validMask))));
fprintf('ANN+Dijkstra%7.3f s      %.1f%% reduction\n', mean(TFLSO_S3(validMask)), ...
    100*(1 - mean(TFLSO_S3(validMask))/mean(TFLSO_S0(validMask))));

%% ================================================================
%% 5. Figure 1 — T_FLSO vs GS2 Longitude
%% ================================================================
lonVec = longitudeGs2_all;
valid  = ~isnan(TFLSO_S0);

fig1 = figure('Color','w','Position',[100 100 760 420]);
hold on; box on; grid on;

plot(lonVec(valid), TFLSO_S0(valid), 'k-s', ...
    'LineWidth',1.8, 'MarkerSize',7, 'MarkerFaceColor','k');
plot(lonVec(valid), TFLSO_S1(valid), 'b-o', ...
    'LineWidth',1.8, 'MarkerSize',7, 'MarkerFaceColor','b');
plot(lonVec(valid), TFLSO_S2(valid), 'g-^', ...
    'LineWidth',1.8, 'MarkerSize',7, 'MarkerFaceColor','g');
plot(lonVec(valid), TFLSO_S3(valid), 'r-d', ...
    'LineWidth',1.8, 'MarkerSize',7, 'MarkerFaceColor','r');

xlabel('GS2 Longitude (°)', 'FontSize',13);
ylabel('T_{FLSO} (s)',        'FontSize',13);
title('FLSO Switch-Over Delay vs. GS2 Longitude', 'FontSize',13, 'FontWeight','bold');
legend({'S0: Baseline (No ANN)', ...
        'S1: ANN-Only', ...
        'S2: ANN + Clustering', ...
        'S3: ANN + Dijkstra'}, ...
    'Location','northeast', 'FontSize',11);
set(gca,'FontSize',12);
xticks(lonVec);
xtickangle(45);

saveas(fig1, fullfile(outDir, 'fig1_TFLSO_vs_lon_caseCFinal.png'));
exportgraphics(fig1, fullfile(outDir, 'fig1_TFLSO_vs_lon_caseCFinal.pdf'), ...
    'ContentType','vector');
fprintf('\nFig 1 saved.\n');

%% ================================================================
%% 6. Figure 2 — Data Loss vs GS2 Longitude
%% ================================================================
fig2 = figure('Color','w','Position',[100 100 760 420]);
hold on; box on; grid on;

plot(lonVec(valid), DataLoss_S0(valid), 'k-s', 'LineWidth',1.8,'MarkerSize',7,'MarkerFaceColor','k');
plot(lonVec(valid), DataLoss_S1(valid), 'b-o', 'LineWidth',1.8,'MarkerSize',7,'MarkerFaceColor','b');
plot(lonVec(valid), DataLoss_S2(valid), 'g-^', 'LineWidth',1.8,'MarkerSize',7,'MarkerFaceColor','g');
plot(lonVec(valid), DataLoss_S3(valid), 'r-d', 'LineWidth',1.8,'MarkerSize',7,'MarkerFaceColor','r');

xlabel('GS2 Longitude (°)',    'FontSize',13);
ylabel('Data Loss (GB/event)', 'FontSize',13);
title('Data Loss During FLSO vs. GS2 Longitude', 'FontSize',13, 'FontWeight','bold');
legend({'S0: Baseline (No ANN)', ...
        'S1: ANN-Only', ...
        'S2: ANN + Clustering', ...
        'S3: ANN + Dijkstra'}, ...
    'Location','northeast', 'FontSize',11);
set(gca,'FontSize',12);
xticks(lonVec);
xtickangle(45);

saveas(fig2, fullfile(outDir, 'fig2_DataLoss_vs_lon_caseCFinal.png'));
exportgraphics(fig2, fullfile(outDir, 'fig2_DataLoss_vs_lon_caseCFinal.pdf'), ...
    'ContentType','vector');
fprintf('Fig 2 saved.\n');

%% ================================================================
%% 7. Figure 3 — Throughput Timeline during FLSO window
%%    (sample scenario: pick GS2 at lon=210° as representative)
%% ================================================================
lonSample = 210;
jSample   = find(longitudeGs2_all == lonSample, 1);
if isempty(jSample)
    [~, jSample] = min(abs(longitudeGs2_all - 210));
    lonSample = longitudeGs2_all(jSample);
end

T_flso_sample = [TFLSO_S0(jSample), TFLSO_S1(jSample), ...
                 TFLSO_S2(jSample), TFLSO_S3(jSample)];

% Timeline window: -10s before failure to max(T_FLSO)+5s after
t_fail  = 0;       % reference: failure occurs at t=0
t_win   = max(T_flso_sample) + 5;
t_axis  = linspace(-10, t_win, 2000);

% Throughput model:
%   Before failure: 100% (feeder link at GS1 active)
%   During FLSO window: 0% (link disrupted)
%   After FLSO resolves: 100% (GS2 link active)
% We model a short ramp-up (0.1s) after switch-over for simplicity.

ramp = 0.1;  % s — link activation ramp

throughput_fn = @(t_ax, T_sw) ...
    (t_ax < t_fail) * 1.0 + ...                             % before failure
    (t_ax >= t_fail & t_ax < T_sw) * 0.0 + ...              % during outage
    (t_ax >= T_sw   & t_ax < T_sw + ramp) .* ...
        ((t_ax(t_ax >= T_sw & t_ax < T_sw + ramp) - T_sw) / ramp) + ...
    (t_ax >= T_sw + ramp) * 1.0;                            % after switch-over

% Evaluate for each scheme
TP_S0 = arrayfun(@(t2) throughput_eval(t2, t_fail, T_flso_sample(1), ramp), t_axis);
TP_S1 = arrayfun(@(t2) throughput_eval(t2, t_fail, T_flso_sample(2), ramp), t_axis);
TP_S2 = arrayfun(@(t2) throughput_eval(t2, t_fail, T_flso_sample(3), ramp), t_axis);
TP_S3 = arrayfun(@(t2) throughput_eval(t2, t_fail, T_flso_sample(4), ramp), t_axis);

fig3 = figure('Color','w','Position',[100 100 820 420]);
hold on; box on; grid on;

plot(t_axis, TP_S0 * link_rate_Gbps, 'k-', 'LineWidth',2.0);
plot(t_axis, TP_S1 * link_rate_Gbps, 'b-', 'LineWidth',2.0);
plot(t_axis, TP_S2 * link_rate_Gbps, 'g-', 'LineWidth',2.0);
plot(t_axis, TP_S3 * link_rate_Gbps, 'r-', 'LineWidth',2.0);

xline(t_fail, 'k--', 'LineWidth',1.2);
text(t_fail+0.3, 9.5, 'Failure', 'FontSize',10, 'Color','k');

% Annotate switch-over points
colors = {'k','b','g','r'};
labels_sw = {'S0','S1','S2','S3'};
for s = 1:4
    xline(T_flso_sample(s), '--', 'Color',colors{s}, 'LineWidth',1.0);
end

xlabel('Time since ISL failure event (s)', 'FontSize',13);
ylabel('Throughput (Gbps)',                'FontSize',13);
title(sprintf('Throughput During FLSO Window  (GS2 lon = %d°)', lonSample), ...
    'FontSize',13, 'FontWeight','bold');
legend({'S0: Baseline (No ANN)', ...
        'S1: ANN-Only', ...
        'S2: ANN + Clustering', ...
        'S3: ANN + Dijkstra'}, ...
    'Location','southeast', 'FontSize',11);
ylim([-0.5, link_rate_Gbps + 1]);
set(gca,'FontSize',12);

saveas(fig3, fullfile(outDir, 'fig3_Throughput_FLSO_caseCFinal.png'));
exportgraphics(fig3, fullfile(outDir, 'fig3_Throughput_FLSO_caseCFinal.pdf'), ...
    'ContentType','vector');
fprintf('Fig 3 saved.\n');

%% ================================================================
%% 8. Figure 4 — Hop count vs GS2 longitude (supplementary)
%% ================================================================
fig4 = figure('Color','w','Position',[100 100 620 360]);
bar(lonVec(valid), N_hops_avg(valid), 0.6, 'FaceColor',[0.4 0.6 0.9]);
hold on; box on; grid on;
xlabel('GS2 Longitude (°)', 'FontSize',13);
ylabel('Average ISL Hops',  'FontSize',13);
title('Minimum ISL Hops from GS1 to GS2 vs. GS2 Longitude','FontSize',13,'FontWeight','bold');
set(gca,'FontSize',12);
xticks(lonVec(valid));
xtickangle(45);

saveas(fig4, fullfile(outDir, 'fig4_Hops_vs_lon_caseCFinal.png'));
exportgraphics(fig4, fullfile(outDir, 'fig4_Hops_vs_lon_caseCFinal.pdf'), ...
    'ContentType','vector');
fprintf('Fig 4 saved.\n');

%% ================================================================
%% 9. Save numerical results
%% ================================================================
flsoResults = struct();
flsoResults.lonGS2        = longitudeGs2_all(:);
flsoResults.TFLSO_S0      = TFLSO_S0;
flsoResults.TFLSO_S1      = TFLSO_S1;
flsoResults.TFLSO_S2      = TFLSO_S2;
flsoResults.TFLSO_S3      = TFLSO_S3;
flsoResults.DataLoss_S0   = DataLoss_S0;
flsoResults.DataLoss_S1   = DataLoss_S1;
flsoResults.DataLoss_S2   = DataLoss_S2;
flsoResults.DataLoss_S3   = DataLoss_S3;
flsoResults.N_events      = N_events_per_GS2;
flsoResults.N_hops_avg    = N_hops_avg;
flsoResults.TPR_ANN       = TPR_ANN;
flsoResults.FPR_ANN       = FPR_ANN;
flsoResults.T_react       = T_react;
flsoResults.T_lead        = T_lead;
flsoResults.T_hop_s       = T_hop;
flsoResults.link_rate_Gbps = link_rate_Gbps;

save(fullfile(outDir, 'flso_results_caseCFinal.mat'), 'flsoResults', '-v7.3');
fprintf('\nNumerical results saved to flso_results_caseCFinal.mat\n');
fprintf('\nDone.\n');

%% ================================================================
%% Local functions
%% ================================================================
function tp = throughput_eval(t, t_fail, T_sw, ramp)
    if t < t_fail
        tp = 1.0;
    elseif t < T_sw
        tp = 0.0;
    elseif t < T_sw + ramp
        tp = (t - T_sw) / ramp;
    else
        tp = 1.0;
    end
end

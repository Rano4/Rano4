%% dati2_caseCFinal.m
% ISL geometry processor — Case C.
% Reads dati1_caseCFinal.mat, computes pairwise distances, K-nearest ISL
% candidates, clearance margins, and derived timing quantities.

clearvars; clc;

%% ---- Load scenario ----------------------------------------------------
outDir = fullfile(fileparts(mfilename('fullpath')), 'caseCFinal');
load(fullfile(outDir, 'dati1_caseCFinal.mat'));

%% ---- Constants --------------------------------------------------------
Rt       = 6371;                    % Earth radius km
c_light  = 299792.458;             % km/s
speed_ISL = 0.7 * c_light;         % ISL signal propagation speed

Kcand    = 8;    % K-nearest ISL candidates per satellite
DmaxNow  = 4500; % km — max allowed current ISL distance
DmaxFut  = 3600; % km — max allowed future ISL distance

%% ---- Orbital chord distance between adjacent satellites ---------------
rOrbit = Rt + OrbitalHeight;
delta_theta = (2*pi) / numSat;                          % angular separation
distance_between_sat = 2 * rOrbit * sin(delta_theta/2); % chord (km)
time_ISL = distance_between_sat / speed_ISL;            % one-hop latency (s)

fprintf('=== Case C Geometry (dati2) ===\n');
fprintf('Orbital radius      : %.1f km\n', rOrbit);
fprintf('Adjacent sat dist   : %.1f km  (chord)\n', distance_between_sat);
fprintf('ISL one-hop latency : %.6f s\n', time_ISL);

%% ---- ECEF conversion helper -------------------------------------------
% For equatorial orbit: lat=0 always, so ECEF = [r*cos(lon), r*sin(lon), 0]
satECEF = @(lonDeg, r) [r.*cosd(lonDeg); r.*sind(lonDeg); zeros(1,numel(lonDeg))];

%% ---- Pre-allocate ISL geometry arrays ---------------------------------
% For each time slot: pairwise distance matrix (numSat x numSat)
% Then select K-nearest candidates per satellite
% Stored: distNow(sat_i, cand_k, t)  clearNow(sat_i, cand_k, t)

distMat   = zeros(numSat, numSat, T);   % full pairwise
candIdx   = zeros(numSat, Kcand, T);    % K-nearest indices
distNow   = zeros(numSat, Kcand, T);    % current distances
distFut   = zeros(numSat, Kcand, T);    % future distances (t+H)
ddNow     = zeros(numSat, Kcand, T);    % distance rate (km/slot)
clrNow    = zeros(numSat, Kcand, T);    % clearance margin now (km)
clrFut    = zeros(numSat, Kcand, T);    % clearance margin future

H = 60;  % look-ahead horizon in slots

fprintf('Computing ISL geometry (T=%d)...\n', T);
for t = 1:T
    pos_t = satECEF(satLonDeg(:,t)', rOrbit);   % 3 x numSat at time t
    t_fut = min(t + H, T);
    pos_f = satECEF(satLonDeg(:,t_fut)', rOrbit); % 3 x numSat at t+H

    % Pairwise distances
    D_now = zeros(numSat);
    D_fut = zeros(numSat);
    for i = 1:numSat
        for j = 1:numSat
            if i ~= j
                D_now(i,j) = norm(pos_t(:,i) - pos_t(:,j));
                D_fut(i,j) = norm(pos_f(:,i) - pos_f(:,j));
            else
                D_now(i,j) = Inf;
                D_fut(i,j) = Inf;
            end
        end
    end
    distMat(:,:,t) = D_now;

    % K-nearest candidates (by current distance, within DmaxNow)
    for i = 1:numSat
        d_row = D_now(i,:);
        d_row(d_row > DmaxNow) = Inf;
        [~, sorted_j] = sort(d_row);
        cands = sorted_j(1:min(Kcand, numSat-1));
        % Pad with self-index if fewer than K valid
        if numel(cands) < Kcand
            cands = [cands, repmat(i, 1, Kcand - numel(cands))]; %#ok<AGROW>
        end
        candIdx(i,:,t) = cands(1:Kcand);

        for k = 1:Kcand
            j = cands(k);
            distNow(i,k,t) = D_now(i,j);
            distFut(i,k,t) = D_fut(i,j);
            ddNow(i,k,t)   = (D_fut(i,j) - D_now(i,j)) / H; % km/slot
            clrNow(i,k,t)  = DmaxNow - D_now(i,j);
            clrFut(i,k,t)  = DmaxFut  - D_fut(i,j);
        end
    end
end

%% ---- Summary statistics -----------------------------------------------
all_dNow = distNow(distNow < DmaxNow & distNow > 0);
fprintf('\ndistNow  mean=%.0f  min=%.0f  max=%.0f km\n', ...
    mean(all_dNow), min(all_dNow), max(all_dNow));

all_clr = clrFut(:);
fprintf('clrFut   mean=%.0f  min=%.0f  max=%.0f km\n', ...
    mean(all_clr), min(all_clr), max(all_clr));

fprintf('ddNow    mean=%.4f  std=%.4f km/slot\n', ...
    mean(ddNow(:)), std(ddNow(:)));

%% ---- Save results -----------------------------------------------------
save(fullfile(outDir, 'dati2_caseCFinal.mat'), ...
    'numSat', 'OrbitalHeight', 'Rt', 'rOrbit', 'T', 'dt_s', 'H', ...
    'Kcand', 'DmaxNow', 'DmaxFut', 'speed_ISL', 'time_ISL', ...
    'distance_between_sat', ...
    'satLonDeg', 'visMat1', 'visMat2', 'longitudeGs1', 'longitudeGs2_all', 'numGS2', ...
    'distMat', 'candIdx', 'distNow', 'distFut', 'ddNow', 'clrNow', 'clrFut');

fprintf('\ndati2_caseCFinal.mat saved to %s\n', outDir);

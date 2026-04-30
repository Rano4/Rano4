%% dati1_caseCFinal.m
% Equatorial LEO constellation scenario builder — Case C (FLSO study).
% 30 satellites, 600 km altitude, inclination = 0 deg.
% GS1 at lon=90 deg; GS2 candidates at lon=150:15:300 deg.
% Uses longitude-only visibility (professor's original convention).

clearvars; clc;

%% ---- Constellation parameters ----------------------------------------
numSat          = 30;
OrbitalHeight   = 600;          % km
InclinationAngle = 0;           % deg  (equatorial)
coverageAngle   = 60;           % deg  half-angle

%% ---- Time parameters ---------------------------------------------------
T      = 3007;   % number of time slots
dt_s   = 10;     % seconds per slot
PI     = pi;

%% ---- Ground stations --------------------------------------------------
latGs1  = 0;   lonGs1  = 90;           % GS1 (feeder uplink)
lonGs2_candidates = 150:15:300;        % GS2 candidate longitudes (deg)

longitudeGs1 = lonGs1;
longitudeGs2_all = lonGs2_candidates;

numGS2 = numel(longitudeGs2_all);

%% ---- Satellite initial longitudes (uniformly spaced) ------------------
initLon = linspace(0, 360 - 360/numSat, numSat);   % deg

%% ---- Orbital angular velocity -----------------------------------------
mu_earth = 3.986e5;                     % km^3/s^2
Re       = 6371;                        % km
r_orbit  = Re + OrbitalHeight;          % km
v_orbit  = sqrt(mu_earth / r_orbit);    % km/s
omega_sat = v_orbit / r_orbit;          % rad/s   (angular velocity)

%% ---- Pre-allocate visibility arrays -----------------------------------
% visMat1(sat, time) — visibility of sat to GS1
visMat1 = false(numSat, T);

% visMat2(sat, GS2_index, time)
visMat2 = false(numSat, numGS2, T);

% Satellite longitudes over time (degrees)
satLonDeg = zeros(numSat, T);

rC_val = coverageRadius(coverageAngle, OrbitalHeight);

%% ---- Simulation loop --------------------------------------------------
for t = 1:T
    elapsed = (t-1) * dt_s;  % seconds
    for i = 1:numSat
        lonDeg_t = mod(initLon(i) + rad2deg(omega_sat * elapsed), 360);
        satLonDeg(i,t) = lonDeg_t;

        phiNow = deg2rad(lonDeg_t);  % satellite longitude in radians

        % Visibility to GS1
        [visMat1(i,t), ~] = visibilityInTime(phiNow, longitudeGs1, coverageAngle, OrbitalHeight);

        % Visibility to each GS2 candidate
        for j = 1:numGS2
            [visMat2(i,j,t), ~] = visibilityInTime(phiNow, longitudeGs2_all(j), coverageAngle, OrbitalHeight);
        end
    end
end

%% ---- Coverage statistics ----------------------------------------------
fprintf('\n=== Case C Scenario (dati1) ===\n');
fprintf('Satellites          : %d\n', numSat);
fprintf('Orbital height      : %d km\n', OrbitalHeight);
fprintf('Coverage radius rC  : %.1f km\n', rC_val);
fprintf('Inclination         : %d deg\n', InclinationAngle);
fprintf('Time slots T        : %d  (dt=%d s, total=%.1f min)\n', T, dt_s, T*dt_s/60);

totalSlots = numSat * T;
gs1Slots   = sum(visMat1(:));
fprintf('\nGS1 visible slots   : %d / %d  (fraction=%.4f)\n', ...
    gs1Slots, totalSlots, gs1Slots/totalSlots);

for j = 1:numGS2
    gs2Slots = sum(sum(visMat2(:,j,:)));
    fprintf('GS2 lon=%3d  visible: %d slots (frac=%.4f)\n', ...
        longitudeGs2_all(j), gs2Slots, gs2Slots/totalSlots);
end

%% ---- Save results -----------------------------------------------------
outDir = fullfile(fileparts(mfilename('fullpath')), 'caseCFinal');
if ~exist(outDir, 'dir'), mkdir(outDir); end

save(fullfile(outDir, 'dati1_caseCFinal.mat'), ...
    'numSat', 'OrbitalHeight', 'InclinationAngle', 'coverageAngle', ...
    'T', 'dt_s', 'rC_val', 'omega_sat', 'r_orbit', ...
    'initLon', 'satLonDeg', ...
    'longitudeGs1', 'longitudeGs2_all', 'numGS2', ...
    'visMat1', 'visMat2');

fprintf('\ndati1_caseCFinal.mat saved to %s\n', outDir);

%% D2: contaminated training - Pd and actual Pfa when target-like outliers leak into S
% m of the K training cells contain an outlier with the SAME steering as the target
% (worst case): X(:,k) += alpha_o*exp(j*phi)*p, phi random per cell and trial.
% Outlier SNR uses the same definition as the CUT: |alpha_o|^2 p'R^-1 p.
% Thresholds are the clean unloaded ones (the detector doesn't know about the outliers).
% S1 = shared scenario (-40/10/20 deg), S2 = S1 + mainlobe jammer at 4 deg (dashed).
% SDET_QUICK=1: 1/20 of the trials, results in results/quick/.

clear; clc; close all;
addpath('../shared-functions');

%% Scenario
quick = strcmp(getenv('SDET_QUICK'), '1');
N = 16; sigma2 = 1;
theta_t = 0; jnr_dB = 35; Pfa = 1e-3;
scen    = {'S1', 'S2'};
theta_j = {[-40 10 20], [-40 4 10 20]};
Klist   = [24 32 48];
SNR_cut = 0:2:24;                  % CUT SNR grid (dB)
m_list  = [1 2];                   % contaminated cells (m = 0 is the clean case)
SNRo_dB = [0 10 20];               % outlier SNR (dB)
M0 = 1e5;                          % trials per threshold and per actual-Pfa point
M1 = 5e3;                          % trials per Pd point
if quick, M0 = M0/20; M1 = M1/20; end
seed = 4200;                       % rng(seed+K): thresholds, seed+1000+K: Pd, seed+2000+K: Pfa
tests = {'AMF', 'Kelly', 'Rao'};
outdir = 'results'; if quick, outdir = 'results/quick'; end

p = steer(N, theta_t);
nS = numel(scen); nK = numel(Klist); nSNR = numel(SNR_cut);
R = cell(1, nS);
for is = 1:nS, R{is} = make_cov(N, sigma2, theta_j{is}, jnr_dB); end

% Cases: 1 = clean, then every (m, outlier SNR) pair
case_m = 0; case_SNRo = NaN;
for m = m_list
    for so = SNRo_dB
        case_m(end+1) = m; case_SNRo(end+1) = so; %#ok<SAGROW>
    end
end
nC = numel(case_m);
case_name = [{'clean'}, compose('m=%d,%ddB', [case_m(2:end); case_SNRo(2:end)].')'];

%% Clean unloaded thresholds (CFAR: one set per K)
eta = nan(nK, 3);
for ik = 1:nK
    K = Klist(ik);
    rng(seed + K); eta(ik, :) = calib(R{1}, N, K, Pfa, [], M0, p);
end

%% Pd vs CUT SNR and actual Pfa, per scenario, K and contamination case
Pd      = nan(nS, nK, nC, nSNR, 3);
Pfa_emp = nan(nS, nK, nC, 3);
for is = 1:nS
    pRp = real(p'*(R{is}\p));
    for ik = 1:nK
        K = Klist(ik);
        fprintf('%s, K = %d ...\n', scen{is}, K);
        for ic = 1:nC
            alpha_o = 0;
            if case_m(ic) > 0, alpha_o = sqrt(10^(case_SNRo(ic)/10) / pRp); end
            rng(seed + 2000 + K);
            Pfa_emp(is, ik, ic, :) = pd_mc(R{is}, K, M0, 0, p, p, [], eta(ik, :), case_m(ic), alpha_o);
            for isnr = 1:nSNR
                alpha = sqrt(10^(SNR_cut(isnr)/10) / pRp);
                rng(seed + 1000 + K);          % same draws for every case and SNR
                Pd(is, ik, ic, isnr, :) = pd_mc(R{is}, K, M1, alpha, p, p, [], eta(ik, :), ...
                                                case_m(ic), alpha_o);
            end
        end
    end
end

%% SNR needed for Pd = 0.9 (interpolated), and its loss vs clean
snr90 = nan(nS, nK, nC, 3);
for is = 1:nS
    for ik = 1:nK
        for ic = 1:nC
            for it = 1:3
                snr90(is, ik, ic, it) = first_crossing(SNR_cut, ...
                    squeeze(Pd(is, ik, ic, :, it)).', 0.9, 'above');
            end
        end
    end
end
loss90 = snr90 - snr90(:, :, 1, :);        % dB, NaN if Pd never reaches 0.9 on the grid

%% Print tables
for is = 1:nS
    for ik = 1:nK
        fprintf('\n%s, K = %d: actual Pfa / 1e-3 under contamination\n', scen{is}, Klist(ik));
        disp(array2table(round(squeeze(Pfa_emp(is, ik, :, :)) / Pfa, 2), ...
            'VariableNames', tests, 'RowNames', case_name));
        fprintf('%s, K = %d: SNR loss (dB) at Pd = 0.9 vs clean (NaN = never reaches 0.9)\n', ...
            scen{is}, Klist(ik));
        disp(array2table(round(squeeze(loss90(is, ik, :, :)), 2), ...
            'VariableNames', tests, 'RowNames', case_name));
    end
end

%% Plots: rows = outlier SNR, columns = m, K = 32; thin lines = clean reference
col = [0 0.447 0.741; 0.85 0.15 0.10; 0.20 0.60 0.20];   % AMF blue, Kelly red, Rao green
mk  = {'o', 's', '^'};
ls  = {'-', '--'};                                        % S1 solid, S2 dashed
ik = find(Klist == 32);
figure('Position', [100 100 1000 900]);
nm = numel(m_list); nso = numel(SNRo_dB);
for iso = 1:nso
    for im = 1:nm
        ic = find(case_m == m_list(im) & case_SNRo == SNRo_dB(iso));
        subplot(nso, nm, (iso-1)*nm + im); hold on; grid on;
        for is = 1:nS
            for it = 1:3
                plot(SNR_cut, squeeze(Pd(is, ik, 1, :, it)), ls{is}, 'Color', [col(it, :) 0.35], ...
                    'LineWidth', 0.8, 'HandleVisibility', 'off');
                plot(SNR_cut, squeeze(Pd(is, ik, ic, :, it)), [ls{is} mk{it}], 'Color', col(it, :), ...
                    'MarkerSize', 4, 'LineWidth', 1.3, 'DisplayName', [tests{it} ' ' scen{is}]);
            end
        end
        ylim([0 1]); xlabel('CUT SNR (dB)'); ylabel('P_D');
        title(sprintf('m = %d outlier(s), outlier SNR = %d dB', m_list(im), SNRo_dB(iso)));
        if iso == 1 && im == nm, legend('Location', 'southeast', 'NumColumns', 2); end
    end
end
sgtitle(sprintf(['P_D with contaminated training, K = 32, clean thresholds ' ...
    '(thin = clean training; S2 dashed)']));

%% Save
if ~exist(outdir, 'dir'), mkdir(outdir); end
saveas(gcf, fullfile(outdir, 'd2_contaminated_training.png'));
save(fullfile(outdir, 'd2_contaminated_training.mat'), 'N', 'sigma2', 'theta_t', 'theta_j', ...
     'jnr_dB', 'Pfa', 'scen', 'Klist', 'SNR_cut', 'm_list', 'SNRo_dB', 'case_m', 'case_SNRo', ...
     'case_name', 'M0', 'M1', 'seed', 'quick', 'tests', 'eta', 'Pd', 'Pfa_emp', 'snr90', 'loss90');

%% D3: does light diagonal loading recover the loss from contaminated training?
% Same contamination model as D2 (m outliers with the target's steering). The CUT SNR
% is fixed at D1's operating point (all tests Pd ~ 0.9 with clean training, K = 32).
% Loading Rhat = S/K + lam*I (load_S). Thresholds are calibrated on CLEAN training with
% the same loading and the same R (loaded tests are not CFAR), per scenario and K.
% Run d1_steering_mismatch.m first (its SNR is used here).
% S1 = shared scenario (-40/10/20 deg), S2 = S1 + mainlobe jammer at 4 deg (dashed).
% SDET_QUICK=1: 1/20 of the trials, results in results/quick/.

clear; clc; close all;
addpath('../shared-functions');

%% Scenario
quick = strcmp(getenv('SDET_QUICK'), '1');
outdir = 'results'; if quick, outdir = 'results/quick'; end
load(fullfile(outdir, 'd1_steering_mismatch.mat'), 'SNR_dB');
N = 16; sigma2 = 1;
theta_t = 0; jnr_dB = 35; Pfa = 1e-3;
scen    = {'S1', 'S2'};
theta_j = {[-40 10 20], [-40 4 10 20]};
Klist   = [24 32 48];
LNR_dB  = 0:2:10;                  % light loading; 'none' = unloaded
m_list  = [1 2];
SNRo_dB = [0 10 20];
M0 = 1e5;                          % trials per threshold and per actual-Pfa point
M1 = 5e3;                          % trials per Pd point
if quick, M0 = M0/20; M1 = M1/20; end
seed = 4300;                       % rng(seed+K): thresholds, seed+1000+K: Pd, seed+2000+K: Pfa
tests = {'AMF', 'Kelly', 'Rao'};

p = steer(N, theta_t);
nS = numel(scen); nK = numel(Klist); nL = numel(LNR_dB); nT = nL + 1;   % transform 1 = none
lnr_label = [{'none'}, compose('%d', LNR_dB)];
R = cell(1, nS);
for is = 1:nS, R{is} = make_cov(N, sigma2, theta_j{is}, jnr_dB); end

case_m = 0; case_SNRo = NaN;
for m = m_list
    for so = SNRo_dB
        case_m(end+1) = m; case_SNRo(end+1) = so; %#ok<SAGROW>
    end
end
nC = numel(case_m);
case_name = [{'clean'}, compose('m=%d,%ddB', [case_m(2:end); case_SNRo(2:end)].')'];

%% Thresholds (clean training) and Pd / actual Pfa under contamination
eta     = nan(nS, nK, nT, 3);
Pd      = nan(nS, nK, nC, nT, 3);
Pfa_emp = nan(nS, nK, nC, nT, 3);
for is = 1:nS
    pRp = real(p'*(R{is}\p));
    alpha = sqrt(10^(SNR_dB/10) / pRp);
    for ik = 1:nK
        K = Klist(ik);
        fprintf('%s, K = %d ...\n', scen{is}, K);
        Tr = cell(1, nT);
        Tr{1} = @(S) S;
        for il = 1:nL, Tr{il+1} = @(S) load_S(S, K, LNR_dB(il), sigma2); end
        for it = 1:nT
            rng(seed + K);                     % same draws for every loading level
            eta(is, ik, it, :) = calib(R{is}, N, K, Pfa, Tr{it}, M0, p);
        end
        e = squeeze(eta(is, ik, :, :));        % nT x 3
        for ic = 1:nC
            alpha_o = 0;
            if case_m(ic) > 0, alpha_o = sqrt(10^(case_SNRo(ic)/10) / pRp); end
            rng(seed + 1000 + K);
            Pd(is, ik, ic, :, :) = pd_mc(R{is}, K, M1, alpha, p, p, Tr, e, case_m(ic), alpha_o);
            rng(seed + 2000 + K);
            Pfa_emp(is, ik, ic, :, :) = pd_mc(R{is}, K, M0, 0, p, p, Tr, e, case_m(ic), alpha_o);
        end
    end
end

%% Best light loading per case vs unloaded
[Pd_best, il_best] = max(Pd(:, :, :, 2:end, :), [], 4);   % over LNR only
Pd_best = squeeze(Pd_best); LNR_best = LNR_dB(squeeze(il_best));
Pd_none = squeeze(Pd(:, :, :, 1, :));                     % nS x nK x nC x 3

%% Print tables
for is = 1:nS
    for ik = 1:nK
        for it = 1:3
            fprintf('\n%s, K = %d, %s: Pd at SNR = %.1f dB vs loading (LNR dB)\n', ...
                scen{is}, Klist(ik), tests{it}, SNR_dB);
            disp(array2table(round(squeeze(Pd(is, ik, :, :, it)), 3), ...
                'VariableNames', lnr_label, 'RowNames', case_name));
            fprintf('  actual Pfa / 1e-3:\n');
            disp(array2table(round(squeeze(Pfa_emp(is, ik, :, :, it)) / Pfa, 2), ...
                'VariableNames', lnr_label, 'RowNames', case_name));
        end
    end
end

%% Plots: Pd vs loading per (m, outlier SNR), K = 32; thin lines = clean training
col = [0 0.447 0.741; 0.85 0.15 0.10; 0.20 0.60 0.20];   % AMF blue, Kelly red, Rao green
mk  = {'o', 's', '^'};
ls  = {'-', '--'};                                        % S1 solid, S2 dashed
xv  = [-2, LNR_dB];                                       % 'none' plotted at x = -2
ik = find(Klist == 32);
nm = numel(m_list); nso = numel(SNRo_dB);
figure('Position', [100 100 1250 700]);
for im = 1:nm
    for iso = 1:nso
        ic = find(case_m == m_list(im) & case_SNRo == SNRo_dB(iso));
        subplot(nm, nso, (im-1)*nso + iso); hold on; grid on;
        for is = 1:nS
            for it = 1:3
                plot(xv, squeeze(Pd(is, ik, 1, :, it)), ls{is}, 'Color', [col(it, :) 0.35], ...
                    'LineWidth', 0.8, 'HandleVisibility', 'off');
                plot(xv, squeeze(Pd(is, ik, ic, :, it)), [ls{is} mk{it}], 'Color', col(it, :), ...
                    'MarkerSize', 4, 'LineWidth', 1.3, 'DisplayName', [tests{it} ' ' scen{is}]);
            end
        end
        xline(-1, 'k:', 'HandleVisibility', 'off');
        xticks(xv); xticklabels(lnr_label); xlim([-2.5 LNR_dB(end) + 0.5]);
        ylim([0 1]); xlabel('LNR (dB)'); ylabel('P_D');
        title(sprintf('m = %d, outlier SNR = %d dB', m_list(im), SNRo_dB(iso)));
        if im == 1 && iso == nso, legend('Location', 'southeast', 'NumColumns', 2); end
    end
end
sgtitle(sprintf(['P_D vs light loading with contaminated training, K = 32, SNR = %.1f dB ' ...
    '(thin = clean training; S2 dashed)'], SNR_dB));

%% Save
if ~exist(outdir, 'dir'), mkdir(outdir); end
saveas(gcf, fullfile(outdir, 'd3_contamination_loading.png'));
save(fullfile(outdir, 'd3_contamination_loading.mat'), 'N', 'sigma2', 'theta_t', 'theta_j', ...
     'jnr_dB', 'Pfa', 'scen', 'Klist', 'LNR_dB', 'lnr_label', 'm_list', 'SNRo_dB', 'case_m', ...
     'case_SNRo', 'case_name', 'SNR_dB', 'M0', 'M1', 'seed', 'quick', 'tests', 'eta', 'Pd', ...
     'Pfa_emp', 'Pd_best', 'LNR_best', 'Pd_none');

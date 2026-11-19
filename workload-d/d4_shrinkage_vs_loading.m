%% D4: shrinkage vs diagonal loading across K (including K < N)
% Methods, all at sum scale and each with its own calibrated threshold (same R, same
% transform; loading and shrinkage break CFAR):
%   plain S        - only for K > N (N/A for K <= N)
%   best loading   - load_S, LNR swept over 0:2:20 dB, best per K, test and scenario
%   best fixed rho - fixed_shrink, rho swept over 0:0.1:1 plus log-spaced 1e-4 ... 0.05,
%                    best per K, test and scenario (rho = 0 is plain S, skipped for K <= N)
%   Ledoit-Wolf    - lw_shrink, data-driven rho, no tuning
% Shrinking toward (trace(Rhat)/N)*I is loading with lam = rho/(1-rho)*trace(Rhat)/N,
% and with 35 dB jammers trace(R)/N ~ 1e4, so rho = 0.1 already means ~30 dB of loading.
% The small-rho points let fixed shrinkage reach the light-loading range;
% lnr_equiv is the loading (dB re sigma2) each rho corresponds to on the true R.
% CUT SNR: clairvoyant matched filter Pd = 0.9 (10.8 dB, as in Workload C).
% "Best" picks the max of many noisy Pd estimates, which biases it up by roughly one
% standard error (~0.006 at Pd ~ 0.9 with 5e3 trials).
% S1 = shared scenario (-40/10/20 deg), S2 = S1 + mainlobe jammer at 4 deg (dashed).
% SDET_QUICK=1: 1/20 of the trials, results in results/quick/.

clear; clc; close all;
addpath('../shared-functions');

%% Scenario
quick = strcmp(getenv('SDET_QUICK'), '1');
outdir = 'results'; if quick, outdir = 'results/quick'; end
N = 16; sigma2 = 1;
theta_t = 0; jnr_dB = 35; Pfa = 1e-3;
SNR_dB = snr_clair(0.9, Pfa);      % clairvoyant Pd = 0.9
fprintf('CUT SNR (clairvoyant Pd = 0.9): %.2f dB\n', SNR_dB);
scen    = {'S1', 'S2'};
theta_j = {[-40 10 20], [-40 4 10 20]};
Klist   = [8 12 16 24 32 48];
LNR_dB  = 0:2:20;
rho_fs  = unique([0:0.1:1, kron(10.^(-4:-2), [1 2 5])]);   % 0, 1e-4 ... 5e-2, 0.1 ... 1
M0 = 1e5;                          % trials per threshold
M1 = 5e3;                          % trials per Pd point (and LW rho samples per K)
if quick, M0 = M0/20; M1 = M1/20; end
seed = 4400;                       % rng(seed+K): thresholds, seed+1000+K: Pd, seed+3000+K: LW rho
tests   = {'AMF', 'Kelly', 'Rao'};
methods = {'plain S', 'best loading', 'best fixed shrinkage', 'Ledoit-Wolf'};

p = steer(N, theta_t);
nS = numel(scen); nK = numel(Klist); nL = numel(LNR_dB); nR = numel(rho_fs);
R = cell(1, nS);
for is = 1:nS, R{is} = make_cov(N, sigma2, theta_j{is}, jnr_dB); end
lnr_equiv = nan(nS, nR);           % loading (dB re sigma2) equivalent to each rho, on true R
for is = 1:nS
    lnr_equiv(is, :) = 10*log10(rho_fs ./ (1 - rho_fs) * real(trace(R{is}))/N / sigma2);
end

%% Thresholds and Pd for every method setting
Pd_plain = nan(nS, nK, 3);      eta_plain = nan(nS, nK, 3);
Pd_load  = nan(nS, nK, nL, 3);  eta_load  = nan(nS, nK, nL, 3);
Pd_fs    = nan(nS, nK, nR, 3);  eta_fs    = nan(nS, nK, nR, 3);
Pd_lw    = nan(nS, nK, 3);      eta_lw    = nan(nS, nK, 3);
rho_lw   = nan(nS, nK, M1);     % LW rho on target-free training sets
for is = 1:nS
    alpha = sqrt(10^(SNR_dB/10) / real(p'*(R{is}\p)));
    for ik = 1:nK
        K = Klist(ik);
        fprintf('%s, K = %d ...\n', scen{is}, K);
        % list of transforms: (method, index into its grid, handle)
        Tr = {}; tag = zeros(0, 2);
        if K > N, Tr{end+1} = @(S) S; tag(end+1, :) = [1 1]; end
        for il = 1:nL
            Tr{end+1} = @(S) load_S(S, K, LNR_dB(il), sigma2); tag(end+1, :) = [2 il]; %#ok<SAGROW>
        end
        for ir = 1:nR
            if rho_fs(ir) == 0 && K <= N, continue; end
            Tr{end+1} = @(S) fixed_shrink(S, K, rho_fs(ir)); tag(end+1, :) = [3 ir]; %#ok<SAGROW>
        end
        Tr{end+1} = @(S, X) lw_shrink(X); tag(end+1, :) = [4 1];
        nT = numel(Tr);

        e = nan(nT, 3);
        for it = 1:nT
            rng(seed + K);                     % same noise-only draws for every setting
            e(it, :) = calib(R{is}, N, K, Pfa, Tr{it}, M0, p);
        end
        rng(seed + 1000 + K);                  % same signal draws for every setting
        pd = pd_mc(R{is}, K, M1, alpha, p, p, Tr, e);

        for it = 1:nT
            j = tag(it, 2);
            switch tag(it, 1)
                case 1, Pd_plain(is, ik, :) = pd(it, :);   eta_plain(is, ik, :) = e(it, :);
                case 2, Pd_load(is, ik, j, :) = pd(it, :); eta_load(is, ik, j, :) = e(it, :);
                case 3, Pd_fs(is, ik, j, :) = pd(it, :);   eta_fs(is, ik, j, :) = e(it, :);
                case 4, Pd_lw(is, ik, :) = pd(it, :);      eta_lw(is, ik, :) = e(it, :);
            end
        end

        rng(seed + 3000 + K);
        [~, X] = gen_data(R{is}, K, M1);
        [~, rho] = lw_shrink(X);
        rho_lw(is, ik, :) = rho;
    end
end

%% Best tuned settings (per scenario, K and test)
[Pd_bload, i_l] = max(Pd_load, [], 3);  Pd_bload = squeeze(Pd_bload);  LNR_best = LNR_dB(squeeze(i_l));
[Pd_bfs,   i_r] = max(Pd_fs,   [], 3);  Pd_bfs   = squeeze(Pd_bfs);    rho_best = rho_fs(squeeze(i_r));
i_r = squeeze(i_r); rho_best_lnr = nan(size(i_r));      % loading equivalent of the best rho
for is = 1:nS, rho_best_lnr(is, :, :) = reshape(lnr_equiv(is, i_r(is, :, :)), 1, nK, 3); end
Pd_tuned = max(Pd_bload, Pd_bfs);
gap_lw   = Pd_lw - Pd_tuned;                 % < 0: LW below the tuned optimum
rho_lw_mean = mean(rho_lw, 3);
Pd_method = cat(4, Pd_plain, Pd_bload, Pd_bfs, Pd_lw);   % nS x nK x 3 x 4

%% Print tables
for is = 1:nS
    for it = 1:3
        fprintf('\n%s, %s: Pd at SNR = %.1f dB\n', scen{is}, tests{it}, SNR_dB);
        Tb = table(round(Pd_plain(is, :, it).', 3), round(Pd_bload(is, :, it).', 3), ...
            LNR_best(is, :, it).', round(Pd_bfs(is, :, it).', 3), rho_best(is, :, it).', ...
            round(rho_best_lnr(is, :, it).', 1), ...
            round(Pd_lw(is, :, it).', 3), round(rho_lw_mean(is, :).', 3), round(gap_lw(is, :, it).', 3), ...
            'VariableNames', {'plain', 'best_load', 'LNR', 'best_fs', 'rho', 'rho_as_LNR', 'LW', ...
                              'mean_rho_LW', 'LW_gap'}, ...
            'RowNames', compose('K=%d', Klist));
        disp(Tb);
    end
end

%% Plots: Pd vs K per method (top 4 panels), LW rho distribution per K (bottom right)
col = [0 0.447 0.741; 0.85 0.15 0.10; 0.20 0.60 0.20];   % AMF blue, Kelly red, Rao green
mk  = {'o', 's', '^'};
ls  = {'-', '--'};                                        % S1 solid, S2 dashed
figure('Position', [100 100 1250 750]);
for imeth = 1:4
    subplot(2, 3, imeth); hold on; grid on;
    for is = 1:nS
        for it = 1:3
            plot(Klist, Pd_method(is, :, it, imeth), [ls{is} mk{it}], 'Color', col(it, :), ...
                'MarkerSize', 5, 'LineWidth', 1.3, 'DisplayName', [tests{it} ' ' scen{is}]);
        end
    end
    xline(N, 'k:', 'K = N', 'HandleVisibility', 'off');
    set(gca, 'XScale', 'log'); xticks(Klist); xlim([7 52]);
    ylim([0 1]); xlabel('K'); ylabel('P_D');
    title(methods{imeth});
    if imeth == 1, legend('Location', 'southeast', 'NumColumns', 2); end
end
edges = 0:0.02:1;
for is = 1:nS
    subplot(2, 3, 4 + is); hold on; grid on;
    for ik = 1:nK
        histogram(squeeze(rho_lw(is, ik, :)), edges, 'Normalization', 'probability', ...
            'DisplayStyle', 'stairs', 'LineWidth', 1.3, 'DisplayName', sprintf('K = %d', Klist(ik)));
    end
    xlabel('\rho (Ledoit-Wolf)'); ylabel('fraction of trials');
    title(sprintf('Ledoit-Wolf \\rho, %s', scen{is}));
    legend('Location', 'northeast');
end
sgtitle(sprintf(['Shrinkage vs loading, SNR = %.1f dB (clairvoyant P_D = 0.9), P_{FA} = 10^{-3} ' ...
    '(best = tuned per K and test; S2 dashed)'], SNR_dB));

%% Save
if ~exist(outdir, 'dir'), mkdir(outdir); end
saveas(gcf, fullfile(outdir, 'd4_shrinkage_vs_loading.png'));
save(fullfile(outdir, 'd4_shrinkage_vs_loading.mat'), 'N', 'sigma2', 'theta_t', 'theta_j', ...
     'jnr_dB', 'Pfa', 'scen', 'Klist', 'LNR_dB', 'rho_fs', 'SNR_dB', 'M0', 'M1', 'seed', 'quick', ...
     'tests', 'methods', 'Pd_plain', 'eta_plain', 'Pd_load', 'eta_load', 'Pd_fs', 'eta_fs', ...
     'Pd_lw', 'eta_lw', 'rho_lw', 'Pd_bload', 'LNR_best', 'Pd_bfs', 'rho_best', 'Pd_tuned', ...
     'lnr_equiv', 'rho_best_lnr', ...
     'gap_lw', 'rho_lw_mean', 'Pd_method');

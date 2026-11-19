%% D1: steering mismatch - Pd vs pointing error for AMF, Kelly and Rao
% The detector assumes the target is at theta_t = 0 deg (p = steer(N, 0)), but it
% arrives from a_true = steer(N, dtheta). SNR is the clairvoyant output SINR along the
% TRUE direction, |alpha|^2 a_true' R^-1 a_true, so it is the same at every dtheta.
% Thresholds are the usual unloaded ones (CFAR, one set per K).
% Two jammer scenarios: S1 = shared scenario (-40/10/20 deg), S2 = S1 plus a mainlobe
% jammer at 4 deg. S2 is dashed in the plots.
% Set the environment variable SDET_QUICK=1 for a fast run with 1/20 of the trials
% (results then go to results/quick/).

clear; clc; close all;
addpath('../shared-functions');

%% Scenario
quick = strcmp(getenv('SDET_QUICK'), '1');
N = 16; sigma2 = 1;
theta_t = 0; jnr_dB = 35; Pfa = 1e-3;
scen    = {'S1', 'S2'};
theta_j = {[-40 10 20], [-40 4 10 20]};
Klist   = [24 32 48];
dtheta  = 0:0.5:7;                 % first null of the beampattern is ~7.2 deg
SNR_grid = 8:0.5:20;               % dB, for choosing the operating SNR
M0 = 1e5;                          % noise-only trials per threshold
M1 = 5e3;                          % signal trials per Pd point
if quick, M0 = M0/20; M1 = M1/20; end
seed = 4100;                       % rng(seed + K): thresholds; rng(seed + 1000 + K): Pd
tests = {'AMF', 'Kelly', 'Rao'};
outdir = 'results'; if quick, outdir = 'results/quick'; end

p = steer(N, theta_t);
nS = numel(scen); nK = numel(Klist); nD = numel(dtheta);
R = cell(1, nS);
for is = 1:nS, R{is} = make_cov(N, sigma2, theta_j{is}, jnr_dB); end

%% Thresholds: unloaded tests are CFAR (depend only on N, K, Pfa), so one set per K
eta = nan(nK, 3);
for ik = 1:nK
    K = Klist(ik);
    rng(seed + K); eta(ik, :) = calib(R{1}, N, K, Pfa, [], M0, p);
end
fprintf('Unloaded Kelly threshold, Monte Carlo vs theory:\n');
for ik = 1:nK
    K = Klist(ik);
    fprintf('  K = %2d: %.4f vs %.4f\n', K, eta(ik, 2), 1 - Pfa^(1/(K - N + 1)));
end

%% Operating SNR: smallest grid value where all three tests reach Pd >= 0.9
% at dtheta = 0, K = 32 (S1). Matched unloaded Pd depends only on SINR, N and K,
% so the same SNR serves S2 (checked below via the dtheta = 0 column).
ik32 = find(Klist == 32);
Pd_sweep = nan(numel(SNR_grid), 3);
for i = 1:numel(SNR_grid)
    alpha = sqrt(10^(SNR_grid(i)/10) / real(p'*(R{1}\p)));
    rng(seed + 500); Pd_sweep(i, :) = pd_mc(R{1}, 32, M1, alpha, p, p, [], eta(ik32, :));
end
i0 = find(all(Pd_sweep >= 0.9, 2), 1);
if isempty(i0), error('No SNR in the grid gives Pd >= 0.9 for all tests'); end
SNR_dB = SNR_grid(i0); SNR = 10^(SNR_dB/10);
fprintf('\nOperating SNR (all tests Pd >= 0.9 at dtheta = 0, K = 32): %.1f dB\n', SNR_dB);
fprintf('  Pd there: AMF %.3f, Kelly %.3f, Rao %.3f\n', Pd_sweep(i0, :));

%% Mismatch sweep
Pd   = nan(nS, nK, nD, 3);         % scenario x K x dtheta x test
cos2 = nan(nS, nD);                % |p'R^-1 a|^2 / ((p'R^-1 p)(a'R^-1 a))
for is = 1:nS
    fprintf('%s ...\n', scen{is});
    for id = 1:nD
        a = steer(N, dtheta(id));
        cos2(is, id) = abs(p'*(R{is}\a))^2 / (real(p'*(R{is}\p)) * real(a'*(R{is}\a)));
        alpha = sqrt(SNR / real(a'*(R{is}\a)));
        for ik = 1:nK
            K = Klist(ik);
            rng(seed + 1000 + K);          % same draws for every dtheta
            Pd(is, ik, id, :) = pd_mc(R{is}, K, M1, alpha, a, p, [], eta(ik, :));
        end
    end
end

%% Where does Pd fall below 0.5?
d50 = nan(nS, nK, 3);
for is = 1:nS
    for ik = 1:nK
        for it = 1:3
            d50(is, ik, it) = first_crossing(dtheta, squeeze(Pd(is, ik, :, it)).', 0.5, 'below');
        end
    end
end

%% Print tables
show = 1:2:nD;                     % dtheta = 0, 1, ..., 7
for is = 1:nS
    for it = 1:3
        fprintf('\nPd, %s, %s (SNR = %.1f dB):\n', tests{it}, scen{is}, SNR_dB);
        Tb = array2table(round(squeeze(Pd(is, :, show, it)), 3), ...
            'VariableNames', compose('dth_%g', dtheta(show)), 'RowNames', compose('K=%d', Klist));
        disp(Tb);
    end
end
for is = 1:nS
    fprintf('\ndtheta (deg) where Pd drops below 0.5, %s:\n', scen{is});
    disp(array2table(round(squeeze(d50(is, :, :)), 2), 'VariableNames', tests, ...
        'RowNames', compose('K=%d', Klist)));
end
fprintf('cos^2 at dtheta = 0:1:7 deg:\n');
disp(array2table(round(cos2(:, show), 3), 'VariableNames', compose('dth_%g', dtheta(show)), ...
    'RowNames', scen));

%% Plots: row 1 Pd vs dtheta, row 2 Pd vs cos^2; one column per K
col = [0 0.447 0.741; 0.85 0.15 0.10; 0.20 0.60 0.20];   % AMF blue, Kelly red, Rao green
mk  = {'o', 's', '^'};
ls  = {'-', '--'};                                        % S1 solid, S2 dashed
figure('Position', [100 100 1250 700]);
for ik = 1:nK
    for row = 1:2
        subplot(2, nK, (row-1)*nK + ik); hold on; grid on;
        for is = 1:nS
            for it = 1:3
                yv = squeeze(Pd(is, ik, :, it)).';
                if row == 1, xv = dtheta;
                else, [xv, o] = sort(cos2(is, :)); yv = yv(o);   % cos^2 is not monotonic in S2
                end
                plot(xv, yv, [ls{is} mk{it}], 'Color', col(it, :), ...
                    'MarkerSize', 4, 'LineWidth', 1.3, 'DisplayName', [tests{it} ' ' scen{is}]);
            end
        end
        yline(0.5, ':', 'HandleVisibility', 'off');
        ylim([0 1]); ylabel('P_D');
        if row == 1
            xlabel('\Delta\theta (deg)'); xlim(dtheta([1 end]));
            title(sprintf('K = %d', Klist(ik)));
        else
            xlabel('cos^2 (mismatch)'); xlim([0 1]);
        end
        if row == 1 && ik == nK, legend('Location', 'southwest', 'NumColumns', 2); end
    end
end
sgtitle(sprintf(['P_D vs steering mismatch, SNR = %.1f dB, P_{FA} = 10^{-3}  ' ...
    '(S1 solid: jammers -40/10/20; S2 dashed: +4 deg mainlobe)'], SNR_dB));

%% Save
if ~exist(outdir, 'dir'), mkdir(outdir); end
saveas(gcf, fullfile(outdir, 'd1_steering_mismatch.png'));
save(fullfile(outdir, 'd1_steering_mismatch.mat'), 'N', 'sigma2', 'theta_t', 'theta_j', 'jnr_dB', ...
     'Pfa', 'scen', 'Klist', 'dtheta', 'SNR_grid', 'Pd_sweep', 'SNR_dB', 'M0', 'M1', 'seed', ...
     'quick', 'tests', 'eta', 'Pd', 'cos2', 'd50');

%% C3: best loading level per test and the gap between the detectors
% Part 1 (no new simulation): from C2's results, the LNR that maximises Pd for each
%   test and K, and the detector gap = max Pd - min Pd across AMF, Kelly, Rao.
% Part 2 (new simulation): SNR each test needs for Pd = 0.9, at its own best LNR,
%   reusing C2's thresholds. Gives the gap in dB, which unlike Pd doesn't compress
%   near 0 or 1, so loaded and unloaded gaps can be compared fairly.

clear; clc; close all;
addpath('../shared-functions');
load('results/c2_pd_vs_loading.mat');    % N, theta_t, theta_j, jnr_dB, Pfa, Klist, LNR_dB,
                                          % SNR_dB, tests, eta, Pd, eta_unl, Pd_unl
sigma2 = 1;
p = steer(N, theta_t);
R = make_cov(N, sigma2, theta_j, jnr_dB);
nK = numel(Klist); nL = numel(LNR_dB);

%% Part 1a: best LNR per test and K
[Pd_best, il_best] = max(Pd, [], 2);      % nK x 1 x 3
Pd_best = squeeze(Pd_best); il_best = squeeze(il_best);
LNR_best = LNR_dB(il_best);               % nK x 3

fprintf('Best LNR (dB) and Pd there, SNR = %.2f dB:\n', SNR_dB);
Tb = table(Klist(:), LNR_best(:,1), Pd_best(:,1), LNR_best(:,2), Pd_best(:,2), LNR_best(:,3), Pd_best(:,3), ...
    'VariableNames', {'K', 'AMF_LNR', 'AMF_Pd', 'Kelly_LNR', 'Kelly_Pd', 'Rao_LNR', 'Rao_Pd'});
disp(Tb);

%% Part 1b: detector gap vs LNR
gap     = max(Pd, [], 3) - min(Pd, [], 3);          % nK x nL
gap_unl = max(Pd_unl, [], 2) - min(Pd_unl, [], 2);  % nK x 1 (NaN for K < N)

fprintf('Detector gap (max - min Pd across the three tests):\n');
show = [1 3 6 9 11];                                 % LNR = 0, 4, 10, 16, 20 dB
Tg = array2table(round([gap_unl, gap(:, show), min(gap, [], 2)], 3), ...
    'VariableNames', [{'unloaded'}, compose('LNR_%d', LNR_dB(show)), {'min_loaded'}], ...
    'RowNames', compose('K=%d', Klist));
disp(Tg);

%% Part 2: SNR needed for Pd = 0.9 at each test's best LNR
SNRgrid_dB = 5:0.5:45;
M1 = 1e4;
SNR90     = nan(nK, 3);      % loaded, at each test's best LNR
SNR90_unl = nan(nK, 3);      % unloaded (K >= N)

for ik = 1:nK
    K = Klist(ik);
    fprintf('K = %d ...\n', K);
    rng(2000 + K);
    [n, X] = gen_data(R, K, M1);                     % noise and training, reused for every SNR
    S = pagemtimes(X, 'none', X, 'ctranspose');

    for it = 1:3
        lam = sigma2*10^(LNR_best(ik, it)/10);
        eta_it = eta(ik, il_best(ik, it), it);
        SNR90(ik, it) = snr_for_pd(n, S + K*lam*eye(N), p, R, it, eta_it, SNRgrid_dB, 0.9);
        if K >= N
            SNR90_unl(ik, it) = snr_for_pd(n, S, p, R, it, eta_unl(ik, it), SNRgrid_dB, 0.9);
        end
    end
end

fprintf('\nSNR (dB) for Pd = 0.9 (clairvoyant: %.2f dB). NaN = above %d dB.\n', SNR_dB, SNRgrid_dB(end));
Ts = array2table(round([SNR90, SNR90_unl], 2), ...
    'VariableNames', [strcat(tests, '_loaded'), strcat(tests, '_unloaded')], ...
    'RowNames', compose('K=%d', Klist));
disp(Ts);
fprintf('Spread across tests (dB): loaded %s | unloaded %s\n', ...
    mat2str(round(max(SNR90, [], 2) - min(SNR90, [], 2), 2).'), ...
    mat2str(round(max(SNR90_unl, [], 2) - min(SNR90_unl, [], 2), 2).'));

%% Plots
markers = {'o-', 's-', '^-', 'd-', 'v-', 'p-'};
figure;
subplot(1, 2, 1); hold on; grid on;
for ik = 1:nK
    plot(LNR_dB, gap(ik, :), markers{ik}, 'LineWidth', 1.5, 'DisplayName', sprintf('K = %d', Klist(ik)));
end
xlabel('LNR (dB)'); ylabel('max P_D - min P_D');
title('Gap between AMF, Kelly and Rao'); legend('Location', 'northeast');

subplot(1, 2, 2); hold on; grid on;
for it = 1:3
    plot(Klist, SNR90(:, it), markers{it}, 'LineWidth', 1.5, 'DisplayName', [tests{it} ' loaded (best LNR)']);
end
set(gca, 'ColorOrderIndex', 1);
for it = 1:3
    plot(Klist, SNR90_unl(:, it), ['--' markers{it}(1)], 'DisplayName', [tests{it} ' unloaded']);
end
yline(SNR_dB, ':', 'clairvoyant', 'HandleVisibility', 'off');
xlabel('training snapshots K'); ylabel('SNR for P_D = 0.9 (dB)');
title('Required SNR at P_{FA} = 10^{-3}'); legend('Location', 'northeast');

%% Save
saveas(gcf, 'results/c3_best_loading_gap.png');
save('results/c3_best_loading_gap.mat', 'Klist', 'LNR_dB', 'tests', 'LNR_best', 'Pd_best', ...
     'gap', 'gap_unl', 'SNRgrid_dB', 'M1', 'SNR90', 'SNR90_unl', 'SNR_dB');

%% Local function
function snr90 = snr_for_pd(n, S, p, R, it, eta, SNRgrid_dB, target)
% Pd of test `it` over an SNR grid (same noise n and covariances S at every SNR),
% then the SNR where Pd first reaches `target` (linear interpolation in dB).
    pd = zeros(size(SNRgrid_dB));
    for is = 1:numel(SNRgrid_dB)
        alpha = sqrt(10^(SNRgrid_dB(is)/10) / real(p'*(R\p)));
        T = detect(n + alpha*p, S, p);
        pd(is) = mean(T(:, it) > eta);
    end
    i = find(pd >= target, 1);
    if isempty(i), snr90 = NaN;
    elseif i == 1, snr90 = SNRgrid_dB(1);
    else, snr90 = interp1(pd([i-1 i]), SNRgrid_dB([i-1 i]), target);
    end
end

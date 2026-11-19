%% C4: the cost in CFAR - false-alarm drift under loading
% Thresholds from C2 were calibrated on the shared R (noise 1, jammers at -40/10/20 deg,
% JNR 35 dB). Here the data come from a different R, one change at a time, and we
% measure the actual Pfa with those same thresholds. The loading lam is fixed in
% absolute terms (LNR relative to the ASSUMED noise power 1), as in a real system.
% Unloaded tests are exactly CFAR, so they should stay at 1e-3 in every scenario.

clear; clc; close all;
addpath('../shared-functions');
load('results/c2_pd_vs_loading.mat');    % N, theta_t, Pfa, Klist, LNR_dB, tests, eta, eta_unl, ...

p = steer(N, theta_t);
Kc   = [8 16 32];                        % K values to test (16 and 32 also have unloaded)
LNRc = 0:4:20;                           % loading levels (subset of C2's grid)
M    = 2e5;                              % noise-only trials per scenario and K

%% Scenarios: one change at a time from the nominal one
sc = struct('name', {}, 'sigma2', {}, 'theta_j', {}, 'jnr_dB', {});
sc(1) = struct('name', 'nominal',          'sigma2', 1,   'theta_j', [-40 10 20], 'jnr_dB', 35);
sc(2) = struct('name', 'noise -3 dB',      'sigma2', 0.5, 'theta_j', [-40 10 20], 'jnr_dB', 35);
sc(3) = struct('name', 'noise +3 dB',      'sigma2', 2,   'theta_j', [-40 10 20], 'jnr_dB', 35);
sc(4) = struct('name', 'JNR 25 dB',        'sigma2', 1,   'theta_j', [-40 10 20], 'jnr_dB', 25);
sc(5) = struct('name', 'JNR 45 dB',        'sigma2', 1,   'theta_j', [-40 10 20], 'jnr_dB', 45);
sc(6) = struct('name', 'angles -30/15/35', 'sigma2', 1,   'theta_j', [-30 15 35], 'jnr_dB', 35);
sc(7) = struct('name', '6 jammers',        'sigma2', 1,   'theta_j', [-60 -40 -20 10 20 45], 'jnr_dB', 35);
nS = numel(sc);
% make_cov scales jammer power with sigma2 (JNR is relative to noise), so for the noise
% scenarios the jammer powers are fixed at their nominal level instead:
nominal_P_dB = 35;

%% Measure actual Pfa
nK = numel(Kc); nL = numel(LNRc);
pfa     = nan(nS, nK, nL, 3);            % loaded
pfa_unl = nan(nS, nK, 3);                % unloaded
for is = 1:nS
    if sc(is).sigma2 ~= 1                % noise changes, jammers stay the same
        R = make_cov(N, 1, sc(is).theta_j, nominal_P_dB) + (sc(is).sigma2 - 1)*eye(N);
    else
        R = make_cov(N, 1, sc(is).theta_j, sc(is).jnr_dB);
    end
    fprintf('%s ...\n', sc(is).name);
    for ik = 1:nK
        K = Kc(ik); iK = find(Klist == K);
        hits = zeros(nL, 3); hits_unl = zeros(1, 3);
        rng(3000 + 10*is + K);
        for m0 = 1:1e4:M                                     % chunks of 1e4 trials
            [x, X] = gen_data(R, K, 1e4);
            S = pagemtimes(X, 'none', X, 'ctranspose');
            for il = 1:nL
                lam = 10^(LNRc(il)/10);                      % fixed: assumes noise power 1
                e = squeeze(eta(iK, LNR_dB == LNRc(il), :)).';
                hits(il, :) = hits(il, :) + sum(detect(x, S + K*lam*eye(N), p) > e, 1);
            end
            if K >= N
                hits_unl = hits_unl + sum(detect(x, S, p) > eta_unl(iK, :), 1);
            end
        end
        pfa(is, ik, :, :) = reshape(hits / M, 1, 1, nL, 3);
        if K >= N, pfa_unl(is, ik, :) = hits_unl / M; end
    end
end

%% Print tables: actual Pfa / nominal (1 = no drift)
for it = 1:3
    for ik = 1:nK
        fprintf('\n%s, K = %d: actual Pfa / 1e-3\n', tests{it}, Kc(ik));
        vals = [squeeze(pfa(:, ik, :, it)), pfa_unl(:, ik, it)] / Pfa;
        Tb = array2table(round(vals, 2), 'RowNames', {sc.name}, ...
            'VariableNames', [compose('LNR_%d', LNRc), {'unloaded'}]);
        disp(Tb);
    end
end

%% Plots: rows = tests, columns = K
markers = {'o-', 's-', '^-', 'd-', 'v-', 'p-', 'h-'};
figure;
for it = 1:3
    for ik = 1:nK
        subplot(3, nK+1, (it-1)*(nK+1) + ik); hold on; grid on;
        for is = 1:nS
            plot(LNRc, squeeze(pfa(is, ik, :, it)), markers{is});
        end
        yline(Pfa, '--');
        set(gca, 'YScale', 'log'); ylim([1e-5 1e-1]); yticks(10.^(-5:-1));
        title(sprintf('%s, K = %d', tests{it}, Kc(ik)));
        if it == 3, xlabel('LNR (dB)'); end
        if ik == 1, ylabel('actual P_{FA}'); end
    end
end
% legend in its own column (same default colour order as the panels)
subplot(3, nK+1, [nK+1, 2*(nK+1)]); hold on;
for is = 1:nS
    plot(nan, nan, markers{is});
end
axis off; legend({sc.name}, 'Location', 'west');
sgtitle('Actual P_{FA} with thresholds set on the nominal R (nominal 10^{-3}, dashed)');

%% Save
saveas(gcf, 'results/c4_cfar_drift.png');
scenarios = {sc.name};
save('results/c4_cfar_drift.mat', 'Kc', 'LNRc', 'M', 'tests', 'scenarios', 'sc', 'pfa', 'pfa_unl', 'Pfa');

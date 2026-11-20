%% C4 (extra): loading that tracks an estimate of the noise power
% Fixed loading (C4) sets lam = LNR * (assumed noise power), so a wrong noise power
% changes the false-alarm rate. Here lam = LNR * sigma2_hat, where sigma2_hat is
% estimated from each trial's own training data: the median of the nonzero eigenvalues
% of S/K. That estimate scales with the noise power, so (Section 5.5 of the notes)
% the drift should largely disappear. Under R -> c*R everything scales together and the
% tests are exactly CFAR. The estimate needs fewer than half of the nonzero eigenvalues
% to be jammers (fails e.g. for 6 jammers with K = 8).
% Thresholds are calibrated with the same tracked loading on the nominal R.

clear; clc; close all;
addpath('../shared-functions');

N = 16; theta_t = 0; Pfa = 1e-3;
p = steer(N, theta_t);
Kc   = [8 16 32];
LNRc = 0:4:20;
M0 = 1e5;                        % calibration trials
M  = 2e5;                        % trials per measured Pfa point
M1 = 2e4;                        % signal trials per Pd point (nominal scenario)
tests = {'AMF', 'Kelly', 'Rao'};

%% Scenarios (as in C4, plus everything +3 dB)
Rj = @(th, P_dB) make_cov(N, 1, th, P_dB) - eye(N);     % jammer part only
sc = struct('name', {}, 'R', {});
sc(1) = struct('name', 'nominal',          'R', Rj([-40 10 20], 35) + eye(N));
sc(2) = struct('name', 'noise -3 dB',      'R', Rj([-40 10 20], 35) + 0.5*eye(N));
sc(3) = struct('name', 'noise +3 dB',      'R', Rj([-40 10 20], 35) + 2*eye(N));
sc(4) = struct('name', 'JNR 25 dB',        'R', Rj([-40 10 20], 25) + eye(N));
sc(5) = struct('name', 'JNR 45 dB',        'R', Rj([-40 10 20], 45) + eye(N));
sc(6) = struct('name', 'angles -30/15/35', 'R', Rj([-30 15 35], 35) + eye(N));
sc(7) = struct('name', '6 jammers',        'R', Rj([-60 -40 -20 10 20 45], 35) + eye(N));
sc(8) = struct('name', 'all +3 dB (2R)',   'R', 2*(Rj([-40 10 20], 35) + eye(N)));
nS = numel(sc); nK = numel(Kc); nL = numel(LNRc);
R0 = sc(1).R;

%% Thresholds on the nominal R, and Pd there (SNR where the clairvoyant Pd = 0.9)
c2 = load('results/c2_pd_vs_loading.mat');            % for SNR_dB and the fixed-loading Pd
alpha = sqrt(10^(c2.SNR_dB/10) / real(p'*(R0\p)));
eta = zeros(nK, nL, 3); Pd = zeros(nK, nL, 3);
for ik = 1:nK
    K = Kc(ik);
    for il = 1:nL
        St = @(S) load_tracked(S, K, LNRc(il));
        rng(K);        eta(ik, il, :) = calib(R0, N, K, Pfa, St, M0, p);
        rng(1000 + K); Pd(ik, il, :)  = pd_mc(R0, K, M1, alpha, p, St, squeeze(eta(ik, il, :)).');
    end
end

%% Actual Pfa in every scenario
pfa = zeros(nS, nK, nL, 3);
for is = 1:nS
    fprintf('%s ...\n', sc(is).name);
    for ik = 1:nK
        K = Kc(ik);
        rng(3000 + 10*is + K);
        hits = zeros(nL, 3);
        for m0 = 1:1e4:M
            [x, X] = gen_data(sc(is).R, K, 1e4);
            S = pagemtimes(X, 'none', X, 'ctranspose');
            for il = 1:nL
                e = squeeze(eta(ik, il, :)).';
                hits(il, :) = hits(il, :) + sum(detect(x, load_tracked(S, K, LNRc(il)), p) > e, 1);
            end
        end
        pfa(is, ik, :, :) = reshape(hits / M, 1, 1, nL, 3);
    end
end

%% Print
for it = 1:3
    for ik = 1:nK
        fprintf('\n%s, K = %d: actual Pfa / 1e-3, tracked loading\n', tests{it}, Kc(ik));
        disp(array2table(round(squeeze(pfa(:, ik, :, it)) / Pfa, 2), 'RowNames', {sc.name}, ...
            'VariableNames', compose('LNR_%d', LNRc)));
    end
end
fprintf('\nPd at %.2f dB, nominal scenario: tracked loading vs fixed loading (C2)\n', c2.SNR_dB);
for ik = 1:nK
    iK = find(c2.Klist == Kc(ik));
    [~, iL] = ismember(LNRc, c2.LNR_dB);
    for it = 1:3
        fprintf('  K = %2d %-5s tracked %s\n                fixed   %s\n', Kc(ik), tests{it}, ...
            mat2str(round(Pd(ik, :, it), 3)), mat2str(round(c2.Pd(iK, iL, it), 3)));
    end
end

%% Plots: rows = tests, columns = K (same layout as c4_cfar_drift)
markers = {'o-', 's-', '^-', 'd-', 'v-', 'p-', 'h-', 'x-'};
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
subplot(3, nK+1, [nK+1, 2*(nK+1)]); hold on;
for is = 1:nS
    plot(nan, nan, markers{is});
end
axis off; legend({sc.name}, 'Location', 'west');
sgtitle('Actual P_{FA}, loading tracks the estimated noise power (nominal 10^{-3}, dashed)');

%% Save
saveas(gcf, 'results/c4_noise_tracking.png');
scenarios = {sc.name};
save('results/c4_noise_tracking.mat', 'Kc', 'LNRc', 'M0', 'M', 'M1', 'tests', 'scenarios', ...
     'eta', 'Pd', 'pfa', 'Pfa');

%% Local functions
function S2 = load_tracked(S, K, lnr_dB)
% S + K*lam*I per page, with lam = 10^(lnr_dB/10) * sigma2_hat and sigma2_hat the
% median of the min(K, N) nonzero eigenvalues of S/K.
    N = size(S, 1);
    ev = sort(real(pageeig(S)) / K, 1, 'descend');       % N x 1 x M
    s2 = median(ev(1:min(K, N), 1, :), 1);              % 1 x 1 x M
    S2 = S + K * 10^(lnr_dB/10) * s2 .* eye(N);
end

function Pd = pd_mc(R, K, M, alpha, p, Strans, eta)
% Fraction of M trials (in chunks) whose statistic exceeds eta, for each test.
    chunk = 1e4; hits = zeros(1, 3);
    for m0 = 1:chunk:M
        m = min(chunk, M - m0 + 1);
        [x, X] = gen_data(R, K, m, alpha, p);
        S = Strans(pagemtimes(X, 'none', X, 'ctranspose'));
        hits = hits + sum(detect(x, S, p) > eta, 1);
    end
    Pd = hits / M;
end

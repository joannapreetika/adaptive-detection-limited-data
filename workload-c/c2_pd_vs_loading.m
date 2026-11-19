%% C2: Pd vs loading level for AMF, Kelly and Rao
% For each K and loading level, set thresholds by Monte Carlo at Pfa = 1e-3 (calib),
% then measure Pd at a fixed SNR. Loading: Rhat = S/K + lam*I, passed as S + K*lam*I.
% The SNR is chosen so that the clairvoyant matched filter (known R) has Pd = 0.9.

clear; clc; close all;
addpath('../shared-functions');

%% Scenario
N = 16; sigma2 = 1;
theta_t = 0; theta_j = [-40 10 20]; jnr_dB = 35;
Pfa = 1e-3;
Klist  = [8 12 16 24 32 48];
LNR_dB = 0:2:20;
M0 = 1e5;                        % noise-only trials per threshold
M1 = 2e4;                        % signal trials per Pd point
tests = {'AMF', 'Kelly', 'Rao'};

p = steer(N, theta_t);
R = make_cov(N, sigma2, theta_j, jnr_dB);

%% Fixed SNR: clairvoyant Pd = 0.9
% Known-R matched filter: T = |p'R^-1 x|^2 / (p'R^-1 p) is Exp(1) under H0, so
% eta = ln(1/Pfa), and Pd = Q1(sqrt(2 SNR), sqrt(2 eta)) (Marcum Q).
eta_c = log(1/Pfa);
SNR = fzero(@(s) pd_clair(s, eta_c) - 0.9, [1 100]);
SNR_dB = 10*log10(SNR);
alpha = sqrt(SNR / real(p'*(R\p)));
fprintf('SNR for clairvoyant Pd = 0.9 at Pfa = %g: %.2f dB\n\n', Pfa, SNR_dB);

%% Sweep K and LNR
nK = numel(Klist); nL = numel(LNR_dB);
eta = nan(nK, nL, 3);  Pd = nan(nK, nL, 3);       % loaded
eta_unl = nan(nK, 3);  Pd_unl = nan(nK, 3);        % unloaded (K >= N only)

for ik = 1:nK
    K = Klist(ik);
    fprintf('K = %d ...\n', K);
    for il = 1:nL
        lam = sigma2*10^(LNR_dB(il)/10);
        Strans = @(S) S + K*lam*eye(N);
        rng(K);        eta(ik, il, :) = calib(R, N, K, Pfa, Strans, M0, p);   % same draws for every LNR
        rng(1000 + K); Pd(ik, il, :)  = pd_mc(R, K, M1, alpha, p, Strans, squeeze(eta(ik, il, :)).');
    end
    if K >= N
        rng(K);        eta_unl(ik, :) = calib(R, N, K, Pfa, [], M0, p);
        rng(1000 + K); Pd_unl(ik, :)  = pd_mc(R, K, M1, alpha, p, @(S) S, eta_unl(ik, :));
    end
end

%% Checks
% 1) Unloaded Kelly threshold vs theory: Pfa = (1 - eta)^(K-N+1)
fprintf('\nUnloaded Kelly threshold, Monte Carlo vs theory:\n');
for ik = find(Klist >= N)
    K = Klist(ik);
    fprintf('  K = %2d: %.4f vs %.4f\n', K, eta_unl(ik, 2), 1 - Pfa^(1/(K - N + 1)));
end
% 2) Actual Pfa of one loaded threshold on fresh noise-only data
ik = find(Klist == 16); il = find(LNR_dB == 10);
rng(999);
pfa_check = pd_mc(R, 16, 2e5, 0, p, @(S) S + 16*10*eye(N), squeeze(eta(ik, il, :)).');
fprintf('Fresh-data Pfa at K = 16, LNR = 10 dB: AMF %.2e, Kelly %.2e, Rao %.2e (nominal %.0e)\n', ...
        pfa_check, Pfa);

%% Print Pd tables
show = [1 3 6 9 11];             % LNR = 0, 4, 10, 16, 20 dB
for it = 1:3
    fprintf('\nPd, %s (SNR = %.1f dB, clairvoyant 0.90):\n', tests{it}, SNR_dB);
    Tb = array2table(round([Pd(:, show, it), Pd_unl(:, it)], 3), ...
        'VariableNames', [compose('LNR_%d', LNR_dB(show)), {'unloaded'}], ...
        'RowNames', compose('K=%d', Klist));
    disp(Tb);
end

%% Plots: one panel per K
markers = {'o-', 's-', '^-'};
figure;
for ik = 1:nK
    subplot(2, 3, ik); hold on; grid on;
    for it = 1:3
        plot(LNR_dB, Pd(ik, :, it), markers{it}, 'LineWidth', 1.5, 'DisplayName', tests{it});
    end
    if Klist(ik) >= N
        set(gca, 'ColorOrderIndex', 1);          % same default colour per test
        for it = 1:3
            plot(LNR_dB([1 end]), Pd_unl(ik, it)*[1 1], '--', ...
                'DisplayName', [tests{it} ' unloaded']);
        end
    end
    yline(0.9, ':', 'clairvoyant', 'HandleVisibility', 'off');
    ylim([0 1]);
    xlabel('LNR (dB)'); ylabel('P_D');
    title(sprintf('K = %d', Klist(ik)));
    if ik == nK, legend('Location', 'southeast'); end
end
sgtitle(sprintf('P_D vs loading, SNR = %.1f dB, P_{FA} = 10^{-3}', SNR_dB));

%% Save
if ~exist('results', 'dir'), mkdir('results'); end
saveas(gcf, 'results/c2_pd_vs_loading.png');
save('results/c2_pd_vs_loading.mat', 'N', 'theta_t', 'theta_j', 'jnr_dB', 'Pfa', 'Klist', 'LNR_dB', ...
     'M0', 'M1', 'SNR_dB', 'tests', 'eta', 'Pd', 'eta_unl', 'Pd_unl');

%% Local functions
function Pd = pd_mc(R, K, M, alpha, p, Strans, eta)
% Fraction of M trials (in chunks) whose statistic exceeds eta, for each test.
% alpha = 0 gives the actual false-alarm rate instead.
    N = numel(p); chunk = 1e4; hits = zeros(1, 3);
    for m0 = 1:chunk:M
        m = min(chunk, M - m0 + 1);
        [x, X] = gen_data(R, K, m, alpha, p);
        S = Strans(pagemtimes(X, 'none', X, 'ctranspose'));
        hits = hits + sum(detect(x, S, p) > eta, 1);
    end
    Pd = hits / M;
end

function pd = pd_clair(snr, eta)
% Pd of the known-R matched filter: integral of the noncentral exponential tail
% (= Marcum Q1(sqrt(2 snr), sqrt(2 eta))), using the scaled Bessel function.
    f = @(t) besseli(0, 2*sqrt(snr*t), 1) .* exp(2*sqrt(snr*t) - t - snr);
    pd = integral(f, eta, Inf);
end

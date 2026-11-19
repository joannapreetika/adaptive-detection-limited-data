%% C5: does the required K drop from about 2N to about 2J with loading?
% For J = 1, 3, 6 jammers (nested sets, JNR 35 dB each), find the smallest K that gives
%  (a) mean SINR loss within 3 dB          (C1's method, no thresholds)
%  (b) Pd >= 0.9 at SNR = SNR90 + 3 dB     (C2's method), where SNR90 = 10.76 dB is the
%      SNR at which the clairvoyant detector has Pd = 0.9. So (b) means "detection loss
%      of at most 3 dB", the detection version of (a).
% Loaded at a fixed LNR = 14 dB (middle of the 12-16 dB range from C2/C3) vs unloaded.

clear; clc; close all;
addpath('../shared-functions');

%% Scenario
N = 16; sigma2 = 1; theta_t = 0; jnr_dB = 35; Pfa = 1e-3;
Jsets = {20, [-40 10 20], [-60 -40 -20 10 20 45]};
Jlist = cellfun(@numel, Jsets);
LNR = 14; lam = sigma2*10^(LNR/10);
tests = {'AMF', 'Kelly', 'Rao'};
p = steer(N, theta_t);

%% (a) SINR loss vs K
M = 5000;
Ksinr = 2:64;
loss_dl  = nan(numel(Jlist), numel(Ksinr));
loss_unl = nan(numel(Jlist), numel(Ksinr));
for ij = 1:numel(Jlist)
    R = make_cov(N, sigma2, Jsets{ij}, jnr_dB);
    sinr_opt = real(p'*(R\p));
    rng(4000 + ij);
    for ik = 1:numel(Ksinr)
        K = Ksinr(ik);
        [~, X] = gen_data(R, K, M);
        S = pagemtimes(X, 'none', X, 'ctranspose');
        W = squeeze(pagemldivide(S + K*lam*eye(N), p));
        loss_dl(ij, ik) = mean(sinr_loss(W, p, R, sinr_opt));
        if K >= N
            W = squeeze(pagemldivide(S, p));
            loss_unl(ij, ik) = mean(sinr_loss(W, p, R, sinr_opt));
        end
    end
end
K3_dl  = nan(1, numel(Jlist)); K3_unl = nan(1, numel(Jlist));
for ij = 1:numel(Jlist)
    K3_dl(ij)  = Ksinr(find(10*log10(loss_dl(ij, :))  >= -3, 1));
    K3_unl(ij) = Ksinr(find(10*log10(loss_unl(ij, :)) >= -3, 1));
end

%% (b) Pd vs K at SNR90 + 3 dB
eta_c = log(1/Pfa);
SNR90 = fzero(@(s) pd_clair(s, eta_c) - 0.9, [1 100]);
SNR = 2*SNR90;                                   % +3 dB
fprintf('SNR90 = %.2f dB, test SNR = %.2f dB (clairvoyant Pd = %.4f)\n\n', ...
        10*log10(SNR90), 10*log10(SNR), pd_clair(SNR, eta_c));

Kdl  = [2 3 4 5 6 8 10 12 14 16 20 24 32];       % loaded
Kunl = [17 20 24 28 32 36 40 48 56 64];          % unloaded (K > N)
M0 = 1e5; M1 = 2e4;
Pd_dl  = nan(numel(Jlist), numel(Kdl), 3);
Pd_unl = nan(numel(Jlist), numel(Kunl), 3);
for ij = 1:numel(Jlist)
    R = make_cov(N, sigma2, Jsets{ij}, jnr_dB);
    alpha = sqrt(SNR / real(p'*(R\p)));
    fprintf('J = %d ...\n', Jlist(ij));
    for ik = 1:numel(Kdl)
        K = Kdl(ik);
        Strans = @(S) S + K*lam*eye(N);
        rng(5000 + 100*ij + K); e = calib(R, N, K, Pfa, Strans, M0, p);
        rng(6000 + 100*ij + K); Pd_dl(ij, ik, :) = pd_mc(R, K, M1, alpha, p, Strans, e);
    end
    for ik = 1:numel(Kunl)
        K = Kunl(ik);
        rng(5000 + 100*ij + K); e = calib(R, N, K, Pfa, [], M0, p);
        rng(6000 + 100*ij + K); Pd_unl(ij, ik, :) = pd_mc(R, K, M1, alpha, p, @(S) S, e);
    end
end

% smallest K with Pd >= 0.9 (linear interpolation between grid points)
K90_dl = nan(numel(Jlist), 3); K90_unl = nan(numel(Jlist), 3);
for ij = 1:numel(Jlist)
    for it = 1:3
        K90_dl(ij, it)  = first_crossing(Kdl,  Pd_dl(ij, :, it),  0.9);
        K90_unl(ij, it) = first_crossing(Kunl, Pd_unl(ij, :, it), 0.9);
    end
end

%% Print
fprintf('\nSmallest K for 3 dB mean SINR loss:\n');
disp(table(Jlist(:), 2*Jlist(:), K3_dl(:), K3_unl(:), repmat(2*N-3, numel(Jlist), 1), ...
    'VariableNames', {'J', 'twoJ', 'loaded', 'unloaded', 'RMB_2N_3'}));
fprintf('Smallest K for Pd = 0.9 at %.2f dB (NaN = not reached on the grid):\n', 10*log10(SNR));
disp(array2table(round([Jlist(:), K90_dl, K90_unl], 1), 'VariableNames', ...
    [{'J'}, strcat(tests, '_loaded'), strcat(tests, '_unloaded')]));
for ij = 1:numel(Jlist)
    fprintf('Pd, J = %d, loaded (K = %s):\n', Jlist(ij), mat2str(Kdl));
    for it = 1:3
        fprintf('  %-5s %s\n', tests{it}, mat2str(round(Pd_dl(ij, :, it), 3)));
    end
end

%% Plots
markers = {'o-', 's-', '^-'};
figure;
subplot(1, 3, 1); hold on; grid on;
for ij = 1:numel(Jlist)
    plot(Ksinr, 10*log10(loss_dl(ij, :)), markers{ij}, 'MarkerIndices', 1:4:numel(Ksinr), ...
        'LineWidth', 1.5, 'DisplayName', sprintf('J = %d, loaded', Jlist(ij)));
end
set(gca, 'ColorOrderIndex', 1);
for ij = 1:numel(Jlist)
    plot(Ksinr, 10*log10(loss_unl(ij, :)), '--', 'DisplayName', sprintf('J = %d, unloaded', Jlist(ij)));
end
yline(-3, ':', 'HandleVisibility', 'off');
ylim([-15 0.5]);
xlabel('training snapshots K'); ylabel('mean SINR loss (dB)');
title(sprintf('SINR loss vs K, LNR = %d dB', LNR)); legend('Location', 'southeast');

% required K: same series order in both panels, so colours match
subplot(1, 3, 2); hold on; grid on;
plot(Jlist, K3_dl, 'd-', 'LineWidth', 2, 'DisplayName', 'SINR within 3 dB');
for it = 1:3
    plot(Jlist, K90_dl(:, it), markers{it}, 'DisplayName', ['P_D 0.9, ' tests{it}]);
end
plot(Jlist, 2*Jlist, ':', 'LineWidth', 2, 'DisplayName', '2J');
xlim([0 7]); xticks(Jlist); ylim([0 20]);
xlabel('number of jammers J'); ylabel('required K');
title(sprintf('Loaded (LNR = %d dB)', LNR)); legend('Location', 'northwest');

subplot(1, 3, 3); hold on; grid on;
plot(Jlist, K3_unl, 'd-', 'LineWidth', 2, 'DisplayName', 'SINR within 3 dB');
for it = 1:3
    plot(Jlist, K90_unl(:, it), markers{it}, 'DisplayName', ['P_D 0.9, ' tests{it}]);
end
yline(2*N-3, ':', '2N-3 (RMB)', 'LineWidth', 2, 'HandleVisibility', 'off');
xlim([0 7]); xticks(Jlist); ylim([0 50]);
xlabel('number of jammers J'); ylabel('required K');
title('Unloaded'); legend('Location', 'south');

%% Save
if ~exist('results', 'dir'), mkdir('results'); end
saveas(gcf, 'results/c5_k_vs_j.png');
save('results/c5_k_vs_j.mat', 'N', 'Jsets', 'Jlist', 'jnr_dB', 'LNR', 'Pfa', 'tests', ...
     'Ksinr', 'loss_dl', 'loss_unl', 'K3_dl', 'K3_unl', ...
     'SNR', 'Kdl', 'Kunl', 'Pd_dl', 'Pd_unl', 'K90_dl', 'K90_unl');

%% Local functions
function rho = sinr_loss(W, p, R, sinr_opt)
% SINR loss factor for each column of W
    rho = abs(p'*W).^2 ./ (real(sum(conj(W).*(R*W), 1)) * sinr_opt);
end

function Pd = pd_mc(R, K, M, alpha, p, Strans, eta)
% Fraction of M trials (in chunks) whose statistic exceeds eta, for each test.
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
% Pd of the known-R matched filter (Marcum Q1(sqrt(2 snr), sqrt(2 eta))).
    f = @(t) besseli(0, 2*sqrt(snr*t), 1) .* exp(2*sqrt(snr*t) - t - snr);
    pd = integral(f, eta, Inf);
end

function k = first_crossing(Kgrid, pd, target)
% Smallest K where pd first reaches target, interpolated linearly between grid points.
    i = find(pd >= target, 1);
    if isempty(i), k = NaN;
    elseif i == 1, k = Kgrid(1);
    else, k = interp1(pd([i-1 i]), Kgrid([i-1 i]), target);
    end
end

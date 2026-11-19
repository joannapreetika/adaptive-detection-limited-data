%% C1: SINR loss vs K, with and without diagonal loading
% Loss factor of the adaptive weights w = Rhat^-1 p:
%     rho = |w'p|^2 / ( (w'Rw) (p'R^-1 p) ),   0 < rho <= 1
% Unloaded: Rhat = S/K          (needs K >= N), RMB: E[rho] = (K-N+2)/(K+1)
% Loaded:   Rhat = S/K + lam*I  (written as S + K*lam*I, same w up to scale)

clear; clc; close all;
addpath('../shared-functions');
rng(2026);

%% Scenario
N = 16; sigma2 = 1;
theta_t = 0; theta_j = [-40 10 20]; jnr_dB = 35;
J = numel(theta_j);
M = 5000;                        % trials per point
Klist = 2:48;
LNR_dB = [0 5 10 15 20];

p = steer(N, theta_t);
R = make_cov(N, sigma2, theta_j, jnr_dB);
sinr_opt = real(p'*(R\p));

%% Loss vs K
loss_unl = nan(1, numel(Klist));
loss_dl  = nan(numel(LNR_dB), numel(Klist));

for ik = 1:numel(Klist)
    K = Klist(ik);
    [~, X] = gen_data(R, K, M);
    S = pagemtimes(X, 'none', X, 'ctranspose');        % N x N x M sample covariances

    if K >= N
        W = squeeze(pagemldivide(S, p));               % N x M weights
        loss_unl(ik) = mean(sinr_loss(W, p, R, sinr_opt));
    end
    for il = 1:numel(LNR_dB)
        lam = sigma2*10^(LNR_dB(il)/10);
        W = squeeze(pagemldivide(S + K*lam*eye(N), p));
        loss_dl(il, ik) = mean(sinr_loss(W, p, R, sinr_opt));
    end
end

rmb = (Klist - N + 2)./(Klist + 1);
rmb(Klist < N) = NaN;

%% Loss vs loading level at a few K
Kfine = [8 16 32 48];
LNRfine_dB = -10:1:30;
loss_fine = zeros(numel(Kfine), numel(LNRfine_dB));

for ik = 1:numel(Kfine)
    K = Kfine(ik);
    [~, X] = gen_data(R, K, M);
    S = pagemtimes(X, 'none', X, 'ctranspose');
    for il = 1:numel(LNRfine_dB)
        lam = sigma2*10^(LNRfine_dB(il)/10);
        W = squeeze(pagemldivide(S + K*lam*eye(N), p));
        loss_fine(ik, il) = mean(sinr_loss(W, p, R, sinr_opt));
    end
end

%% Print summary
fprintf('Smallest K with mean loss within 3 dB (2J = %d, 2N-3 = %d):\n', 2*J, 2*N-3);
fprintf('  unloaded : K = %d\n', Klist(find(10*log10(loss_unl) >= -3, 1)));
for il = 1:numel(LNR_dB)
    fprintf('  LNR %2d dB: K = %d\n', LNR_dB(il), Klist(find(10*log10(loss_dl(il,:)) >= -3, 1)));
end
[best, ibest] = max(loss_fine, [], 2);
fprintf('Best LNR for SINR:\n');
for ik = 1:numel(Kfine)
    fprintf('  K = %2d: LNR = %d dB, loss = %.2f dB\n', Kfine(ik), LNRfine_dB(ibest(ik)), 10*log10(best(ik)));
end

%% Plots
figure;
subplot(1, 2, 1); hold on; grid on;
markers = {'o-', 's-', '^-', 'd-', 'v-'};
for il = 1:numel(LNR_dB)
    plot(Klist, 10*log10(loss_dl(il,:)), markers{il}, 'MarkerIndices', 1:4:numel(Klist), ...
        'DisplayName', sprintf('loaded, LNR = %d dB', LNR_dB(il)));
end
plot(Klist, 10*log10(loss_unl), 'o', 'DisplayName', 'unloaded (sim)');
plot(Klist, 10*log10(rmb), '--', 'LineWidth', 2, 'DisplayName', 'RMB theory');
yline(-3, ':', 'HandleVisibility', 'off');
xline(2*J, ':', '2J', 'HandleVisibility', 'off');
xline(2*N-3, ':', '2N-3', 'HandleVisibility', 'off');
ylim([-15 0.5]);
xlabel('training snapshots K'); ylabel('mean SINR loss (dB)');
title('SINR loss vs K'); legend('Location', 'southeast');

subplot(1, 2, 2); hold on; grid on;
for ik = 1:numel(Kfine)
    plot(LNRfine_dB, 10*log10(loss_fine(ik,:)), markers{ik}, 'MarkerIndices', 1:5:numel(LNRfine_dB), ...
        'DisplayName', sprintf('K = %d', Kfine(ik)));
end
yline(-3, ':', 'HandleVisibility', 'off');
ylim([-15 0.5]);
xlabel('LNR = \lambda/\sigma^2 (dB)'); ylabel('mean SINR loss (dB)');
title('SINR loss vs loading level'); legend('Location', 'southwest');

%% Save
if ~exist('results', 'dir'), mkdir('results'); end
saveas(gcf, 'results/c1_sinr_loss.png');
save('results/c1_sinr_loss.mat', 'N', 'theta_t', 'theta_j', 'jnr_dB', 'M', 'Klist', 'LNR_dB', ...
     'loss_unl', 'loss_dl', 'rmb', 'Kfine', 'LNRfine_dB', 'loss_fine');

%% Local function
function rho = sinr_loss(W, p, R, sinr_opt)
% SINR loss factor for each column of W
    rho = abs(p'*W).^2 ./ (real(sum(conj(W).*(R*W), 1)) * sinr_opt);
end

%% C4 (formula check): Pfa drift under a noise-power mismatch, theory vs simulation
% Thresholds are set with noise power sigma0^2 = 1; the data then have noise power
% sigma^2 (jammers unchanged), and the loading lam stays fixed.
%
% Large-K theory: Rhat -> R + lam*I. For a fixed Rhat the AMF statistic under H0 is
% exponential with mean c = w'Rw / (p'w), w = Rhat^-1 p. A threshold set for c0 = c(sigma0^2)
% then gives
%     actual Pfa = Pfa ^ ( c(sigma0^2) / c(sigma^2) )
% Heavy-loading limit (lam >> sigma^2): c is proportional to sigma^2, so
%     actual Pfa = Pfa ^ ( sigma0^2 / sigma^2 )
% Unloaded (lam = 0): c = 1 for every sigma^2, so no drift.

clear; clc; close all;
addpath('../shared-functions');

%% Scenario
N = 16; theta_t = 0; theta_j = [-40 10 20]; P_dB = 35;   % jammer power fixed (absolute)
Pfa = 1e-3;
Klist = [16 48 128];
LNR_dB = [0 12];
mis_dB = -3:1:3;                 % true noise power relative to assumed, in dB
M0 = 1e5;                        % calibration trials
M  = 2e5;                        % trials per measured Pfa point
tests = {'AMF', 'Kelly', 'Rao'};

p = steer(N, theta_t);
Rj = make_cov(N, 1, theta_j, P_dB) - eye(N);            % jammer part only
Rof = @(s2) Rj + s2*eye(N);                              % R for noise power s2

%% Theory curves
mis_fine = linspace(-3, 3, 61);
pfa_theory = zeros(numel(LNR_dB), numel(mis_fine));
for il = 1:numel(LNR_dB)
    lam = 10^(LNR_dB(il)/10);
    c0 = c_of(Rof(1), lam, p);
    for im = 1:numel(mis_fine)
        pfa_theory(il, im) = Pfa^(c0 / c_of(Rof(10^(mis_fine(im)/10)), lam, p));
    end
end
pfa_heavy = Pfa.^(1 ./ 10.^(mis_fine/10));

%% Simulation
pfa_sim = nan(numel(Klist), numel(LNR_dB), numel(mis_dB), 3);
for ik = 1:numel(Klist)
    K = Klist(ik);
    fprintf('K = %d ...\n', K);
    for il = 1:numel(LNR_dB)
        lam = 10^(LNR_dB(il)/10);
        Strans = @(S) S + K*lam*eye(N);
        rng(7000 + K + il);
        eta = calib(Rof(1), N, K, Pfa, Strans, M0, p);   % thresholds at the assumed noise power
        for im = 1:numel(mis_dB)
            R = Rof(10^(mis_dB(im)/10));
            hits = zeros(1, 3);
            rng(8000 + 10*K + 100*il + im);
            for m0 = 1:1e4:M
                [x, X] = gen_data(R, K, 1e4);
                S = Strans(pagemtimes(X, 'none', X, 'ctranspose'));
                hits = hits + sum(detect(x, S, p) > eta, 1);
            end
            pfa_sim(ik, il, im, :) = hits / M;
        end
    end
end

%% Print
for il = 1:numel(LNR_dB)
    fprintf('\nLNR = %d dB: actual Pfa / 1e-3 (mismatch %s dB)\n', LNR_dB(il), mat2str(mis_dB));
    th = interp1(mis_fine, pfa_theory(il, :), mis_dB) / Pfa;
    fprintf('  %-16s %s\n', 'theory (large K)', mat2str(round(th, 3)));
    fprintf('  %-16s %s\n', 'heavy loading', mat2str(round(Pfa.^(1./10.^(mis_dB/10))/Pfa, 3)));
    for ik = 1:numel(Klist)
        for it = 1:3
            fprintf('  %-16s %s\n', sprintf('%s K=%d', tests{it}, Klist(ik)), ...
                mat2str(round(squeeze(pfa_sim(ik, il, :, it)).' / Pfa, 3)));
        end
    end
end

%% Plots: rows = LNR, columns = tests
markers = {'o', 's', '^'};
figure;
for il = 1:numel(LNR_dB)
    for it = 1:3
        subplot(numel(LNR_dB), 3, (il-1)*3 + it); hold on; grid on;
        for ik = 1:numel(Klist)
            plot(mis_dB, squeeze(pfa_sim(ik, il, :, it)), markers{ik}, 'LineWidth', 1.5, ...
                'DisplayName', sprintf('sim, K = %d', Klist(ik)));
        end
        plot(mis_fine, pfa_theory(il, :), '-', 'LineWidth', 2, 'DisplayName', 'theory, large K');
        plot(mis_fine, pfa_heavy, '--', 'LineWidth', 1.5, 'DisplayName', 'heavy-loading limit');
        yline(Pfa, ':', 'HandleVisibility', 'off');
        set(gca, 'YScale', 'log'); ylim([1e-6 1e-1]); yticks(10.^(-6:-1));
        title(sprintf('%s, LNR = %d dB', tests{it}, LNR_dB(il)));
        if il == numel(LNR_dB), xlabel('true noise power vs assumed (dB)'); end
        if it == 1, ylabel('actual P_{FA}'); end
        if il == 1 && it == 1, legend('Location', 'northwest'); end
    end
end

%% Save
saveas(gcf, 'results/c4_drift_formula.png');
save('results/c4_drift_formula.mat', 'N', 'theta_j', 'P_dB', 'Pfa', 'Klist', 'LNR_dB', 'mis_dB', ...
     'M0', 'M', 'tests', 'mis_fine', 'pfa_theory', 'pfa_heavy', 'pfa_sim');

%% Local function
function c = c_of(R, lam, p)
% Mean of the H0 AMF statistic for Rhat = R + lam*I (large-K limit): w'Rw / p'w
    w = (R + lam*eye(size(R, 1))) \ p;
    c = real(w'*R*w) / real(p'*w);
end

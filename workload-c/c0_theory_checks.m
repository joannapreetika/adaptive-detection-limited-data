%% C0: numerical checks of the Workload C theory
% (a) Loaded SINR loss  E[rho] ~ (K+1-J)/(K+1)   (RMB with N-1 replaced by J)
% (b) Large-K loaded loss vs loading level: where loading starts to fill the jammer nulls
% (c) beta_perp = beta - tA (whitened energy of x orthogonal to p, target-free):
%     unloaded mean (N-1)/(K-N+1) vs simulation, and how much loading shrinks it
% (d) Rao ceiling: Rao = Kelly/(1+beta_perp) < 1/(1+beta_perp), so for ANY SNR
%     Pd_Rao <= P(beta_perp < 1/eta_Rao - 1). Compared with Rao's Pd at 40 dB SNR.
% Uses the saved results of C1, C2 and C5.

clear; clc; close all;
addpath('../shared-functions');
N = 16; theta_t = 0; theta_j = [-40 10 20]; jnr_dB = 35;
p = steer(N, theta_t);
R = make_cov(N, 1, theta_j, jnr_dB);

%% (a) loaded SINR loss formula vs C1 (J = 3) and C5 (J = 1, 3, 6)
c1 = load('results/c1_sinr_loss.mat');
c5 = load('results/c5_k_vs_j.mat');
fprintf('(a) Mean SINR loss (dB), formula (K+1-J)/(K+1) vs simulation\n');
Ks = [4 6 8 12 16 24 32 48];
[~, i1] = ismember(Ks, c1.Klist);
fprintf('  J = 3, C1 LNR 15 dB:\n    K       %s\n    formula %s\n    sim     %s\n', mat2str(Ks), ...
    mat2str(round(10*log10((Ks + 1 - 3)./(Ks + 1)), 2)), ...
    mat2str(round(10*log10(c1.loss_dl(c1.LNR_dB == 15, i1)), 2)));
[~, i5] = ismember(Ks, c5.Ksinr);
for ij = 1:numel(c5.Jlist)
    J = c5.Jlist(ij);
    fprintf('  J = %d, C5 LNR 14 dB:\n    formula %s\n    sim     %s\n', J, ...
        mat2str(round(10*log10(max(Ks + 1 - J, 0)./(Ks + 1)), 2)), ...
        mat2str(round(10*log10(c5.loss_dl(ij, i5)), 2)));
end

%% (b) large-K loaded loss vs LNR (Rhat = R + lam*I), and the jammer eigenvalues
LNR = 0:5:40;
rho_inf = zeros(size(LNR));
sinr_opt = real(p'*(R\p));
for il = 1:numel(LNR)
    w = (R + 10^(LNR(il)/10)*eye(N)) \ p;
    rho_inf(il) = abs(w'*p)^2 / (real(w'*R*w) * sinr_opt);
end
ev = sort(eig(R), 'descend');
fprintf('\n(b) Jammer eigenvalues of R: %s dB\n', mat2str(round(10*log10(ev(1:3)).', 1)));
fprintf('    Large-K loss (dB) at LNR %s:\n    %s\n', mat2str(LNR), mat2str(round(10*log10(rho_inf), 2)));
g = abs(steer(N, theta_j)'*p).^2 / N^2;     % beam pattern of p at the jammer angles
fprintf('    Quiescent beam pattern at the jammers %s deg: %s dB\n', mat2str(theta_j), ...
    mat2str(round(10*log10(g.'), 1)));

%% (c) beta_perp: unloaded theory vs simulation, and loaded
fprintf('\n(c) Mean beta_perp under H0 (unloaded theory (N-1)/(K-N+1))\n');
M = 2e4;
Kb = [8 16 24 48];
for K = Kb
    rng(9000 + K);
    [x, X] = gen_data(R, K, M);
    S = pagemtimes(X, 'none', X, 'ctranspose');
    bl = mean(beta_perp(x, S + K*10^(14/10)*eye(N), p));
    if K > N
        bu = mean(beta_perp(x, S, p));
        fprintf('  K = %2d: unloaded theory %.3f, sim %.3f | loaded (14 dB) sim %.3f\n', ...
                K, (N-1)/(K-N+1), bu, bl);
    else
        fprintf('  K = %2d: unloaded n/a                     | loaded (14 dB) sim %.3f\n', K, bl);
    end
end

%% (d) Rao ceiling vs Rao Pd at very high SNR (C2 thresholds)
c2 = load('results/c2_pd_vs_loading.mat');
fprintf('\n(d) Rao ceiling P(beta_perp < 1/eta - 1) vs Rao Pd at SNR = 40 dB\n');
alpha = sqrt(10^(40/10) / real(p'*(R\p)));
for K = [8 16]
    for lnr = [0 16]
        eta = c2.eta(c2.Klist == K, c2.LNR_dB == lnr, 3);
        rng(9500 + K + lnr);
        [n, X] = gen_data(R, K, M);
        S = pagemtimes(X, 'none', X, 'ctranspose') + K*10^(lnr/10)*eye(N);
        ceil_pd = mean(beta_perp(n, S, p) < 1/eta - 1);
        T = detect(n + alpha*p, S, p);
        fprintf('  K = %2d, LNR = %2d dB: eta_Rao = %.3f, ceiling %.3f, Pd(40 dB) %.3f\n', ...
                K, lnr, eta, ceil_pd, mean(T(:, 3) > eta));
    end
end

%% Local function
function b = beta_perp(x, S, p)
% x'S^-1 x - |p'S^-1 x|^2/(p'S^-1 p) for each column of x (each page of S)
    [N, M] = size(x);
    Z = pagemldivide(S, [reshape(x, N, 1, M), repmat(p, 1, 1, M)]);
    Six = reshape(Z(:, 1, :), N, M); Sip = reshape(Z(:, 2, :), N, M);
    b = real(sum(conj(x).*Six, 1)) - abs(p'*Six).^2 ./ real(p'*Sip);
end

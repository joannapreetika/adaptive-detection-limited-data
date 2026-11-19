function tests = test_workload_d
%TEST_WORKLOAD_D  Unit tests for Workload D. From inside workload-d/:
%   results = runtests('tests'); disp(table(results))
    tests = functiontests(localfunctions);
end

function setupOnce(tc)
    here = fileparts(mfilename('fullpath'));
    tc.TestData.paths = {fullfile(here, '..'), fullfile(here, '..', '..', 'shared-functions')};
    addpath(tc.TestData.paths{:});
    tc.TestData.N = 16;
    tc.TestData.R = make_cov(16, 1, [-40 10 20], 35);      % shared scenario (S1)
    tc.TestData.p = steer(16, 0);
end

function teardownOnce(tc)
    rmpath(tc.TestData.paths{:});
end

%% 1. Rao statistic in detect() = explicit |p'(S+xx')^-1 x|^2 / (p'(S+xx')^-1 p)
function testRaoMatchesExplicit(tc)
    % Random (white) data at 1e-10 as specified. With the 35 dB jammers S is
    % ill-conditioned (cond ~ 1e4-1e5) and both formulas carry ~cond*eps rounding
    % error, so that case is checked at 1e-8.
    N = tc.TestData.N; p = tc.TestData.p;
    cases = {eye(N), 1e-10; tc.TestData.R, 1e-8};
    rng(1);
    for ic = 1:size(cases, 1)
        R = cases{ic, 1}; tol = cases{ic, 2};
        for K = [20 32]
            [x, X] = gen_data(R, K, 50, 0.3, steer(N, 2));
            S0 = pagemtimes(X, 'none', X, 'ctranspose');
            % plain, loaded and LW-shrunk S: the identity holds for any Hermitian PD S
            for S = {S0, load_S(S0, K, 10, 1), lw_shrink(X)}
                T = detect(x, S{1}, p);
                rao = zeros(size(x, 2), 1);
                for m = 1:size(x, 2)
                    A = S{1}(:, :, m) + x(:, m)*x(:, m)';
                    rao(m) = abs(p'*(A\x(:, m)))^2 / real(p'*(A\p));
                end
                tc.verifyEqual(T(:, 3), rao, 'RelTol', tol);
            end
        end
    end
end

%% 2. Kelly is CFAR: Pfa = (1 - eta)^(K-N+1) at K = 32, no transform
function testKellyCfar(tc)
    N = tc.TestData.N; R = tc.TestData.R; p = tc.TestData.p;
    K = 32; Pfa = 1e-3; M = 1e5;
    eta_th = 1 - Pfa^(1/(K - N + 1));
    rng(2);
    rate = pd_mc(R, K, M, 0, p, p, [], [Inf eta_th Inf]);
    sd = sqrt(Pfa*(1 - Pfa)/M);
    tc.verifyLessThan(abs(rate(2) - Pfa), 4*sd, ...
        sprintf('Kelly Pfa %.2e vs nominal %.0e', rate(2), Pfa));
end

%% 3. lw_shrink: Hermitian, PD, sum scale, rho in [0, 1], b2 shortcut = literal sum
function testLedoitWolf(tc)
    N = tc.TestData.N; R = tc.TestData.R;
    rng(3);
    for K = [8 12 32 48]
        [~, X] = gen_data(R, K, 200);
        [S, rho] = lw_shrink(X);
        S0 = pagemtimes(X, 'none', X, 'ctranspose');
        tc.verifyTrue(all(rho >= 0 & rho <= 1));
        for m = 1:size(X, 3)
            Sm = S(:, :, m); S0m = S0(:, :, m);
            tc.verifyLessThan(norm(Sm - Sm', 'fro') / norm(Sm, 'fro'), 1e-12);
            tc.verifyGreaterThan(min(eig((Sm + Sm')/2)), 0);
            tc.verifyEqual(real(trace(Sm)), real(trace(S0m)), 'RelTol', 1e-10);
        end
        % literal Ledoit-Wolf formula on a few pages
        for m = 1:5
            Rh = S0(:, :, m)/K; mu = real(trace(Rh))/N; F = mu*eye(N);
            d2 = norm(Rh - F, 'fro')^2;
            b2bar = 0;
            for k = 1:K
                xk = X(:, k, m);
                b2bar = b2bar + norm(xk*xk' - Rh, 'fro')^2;
            end
            b2 = min(b2bar/K^2, d2);
            tc.verifyEqual(rho(m), b2/d2, 'AbsTol', 1e-10);
            tc.verifyEqual(S(:, :, m), K*((1 - b2/d2)*Rh + (b2/d2)*F), ...
                'AbsTol', 1e-8*norm(S0(:, :, m)));
        end
    end
end

%% 4. Mismatch code path with dtheta = 0 reproduces the matched case exactly
function testZeroMismatchIsMatched(tc)
    N = tc.TestData.N; R = tc.TestData.R; p = tc.TestData.p;
    K = 32; eta = [12 0.3 0.4];
    a0 = steer(N, 0);                                   % what d1 builds at dtheta = 0
    alpha = sqrt(10^(1.2)/real(a0'*(R\a0)));
    rng(4); r_mis = pd_mc(R, K, 2e4, alpha, a0, p, [], eta);
    rng(4); r_mat = pd_mc(R, K, 2e4, alpha, p,  p, [], eta);
    tc.verifyEqual(r_mis, r_mat);
    % and zero contaminated cells reproduce clean training exactly
    rng(4); r_m0 = pd_mc(R, K, 2e4, alpha, p, p, [], eta, 0, 5);
    tc.verifyEqual(r_m0, r_mat);
end

%% 5. calib with a two-argument transform: same thresholds, same random draws
function testCalibTwoArgTransform(tc)
    N = tc.TestData.N; R = tc.TestData.R; p = tc.TestData.p;
    K = 12; M = 2e4;
    rng(5); e1 = calib(R, N, K, 1e-3, @(S) load_S(S, K, 6, 1), M, p);
    rng(5); e2 = calib(R, N, K, 1e-3, @(S, X) load_S(S, K, 6, 1), M, p);
    tc.verifyEqual(e1, e2);
    rng(5); e3 = calib(R, N, K, 1e-3, @(S, X) lw_shrink(X), M, p);
    tc.verifyTrue(all(isfinite(e3) & e3 > 0));
end

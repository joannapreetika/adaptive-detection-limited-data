function eta = calib(R, N, K, Pfa, Strans, M, p)
%CALIB  Monte Carlo thresholds for [AMF, Kelly, Rao] at a given false-alarm rate.
%   eta = calib(R, N, K, Pfa)                 unloaded
%   eta = calib(R, N, K, Pfa, Strans)         Strans is applied to every S, e.g.
%                                             Strans = @(S) S + K*lam*eye(N)
%   eta = calib(R, N, K, Pfa, Strans, M)      M noise-only trials (default 1e5)
%   eta = calib(R, N, K, Pfa, Strans, M, p)   steering vector (default steer(N, 0))
%   eta : 1 x 3 thresholds, the empirical (1 - Pfa) quantile of each statistic under H0.
%
%   Unloaded thresholds depend only on N, K, Pfa. Loaded ones (no longer CFAR) also
%   depend on R and p, so calibrate with the same R and p you detect with.
%   Use M well above 1/Pfa (1e5 for Pfa = 1e-3 gives ~100 exceedances).
%   Random numbers come from the global stream: set rng() before calling to reuse
%   the same draws across calls (e.g. across loading levels).

    if nargin < 5 || isempty(Strans), Strans = @(S) S; end
    if nargin < 6 || isempty(M), M = 1e5; end
    if nargin < 7, p = steer(N, 0); end

    chunk = 1e4;
    T = zeros(M, 3);
    for m0 = 1:chunk:M
        m = min(chunk, M - m0 + 1);
        [x, X] = gen_data(R, K, m);
        S = Strans(pagemtimes(X, 'none', X, 'ctranspose'));
        T(m0:m0+m-1, :) = detect(x, S, p);
    end

    T = sort(T, 1);
    eta = T(ceil((1 - Pfa) * M), :);
end

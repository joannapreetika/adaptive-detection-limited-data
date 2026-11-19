function [S, rho] = lw_shrink(X)
%LW_SHRINK  Ledoit-Wolf (2004) shrinkage toward the scaled identity, complex data.
%   [S, rho] = lw_shrink(X)
%   X   : N x K x M   training snapshots (one set of K per page)
%   S   : N x N x M   K*R_sh  (SUM scale, like S = X*X')
%   rho : 1 x M       shrinkage weight in [0, 1]
%
%   Rh = S/K, mu = trace(Rh)/N, F = mu*I
%   d2    = ||Rh - F||_F^2
%   b2bar = (1/K^2) sum_k ||x_k x_k' - Rh||_F^2,   b2 = min(b2bar, d2)
%   rho   = b2/d2,   R_sh = (1-rho)*Rh + rho*F
%   The sum in b2bar is evaluated as sum_k ||x_k||^4 - K*||Rh||_F^2 (identical, since
%   ||x x'||_F = ||x||^2 and sum_k x_k x_k' = K*Rh), so no N x N x K temporaries.
%   Use with calib as Strans = @(S, X) lw_shrink(X).

    [N, K, M] = size(X);
    S  = pagemtimes(X, 'none', X, 'ctranspose');
    Rv = reshape(S, N*N, M) / K;                       % Rh, one column per page
    mu = real(sum(Rv(1:N+1:end, :), 1)) / N;           % 1 x M
    nR2 = sum(abs(Rv).^2, 1);                          % ||Rh||_F^2
    d2 = nR2 - N*mu.^2;                                % ||Rh - mu I||_F^2
    xn2 = reshape(sum(abs(X).^2, 1), K, M);            % ||x_k||^2
    b2bar = (sum(xn2.^2, 1) - K*nR2) / K^2;
    b2 = min(b2bar, d2);
    rho = b2 ./ d2;
    r3 = reshape(rho, 1, 1, M);
    S = (1 - r3).*S + K*r3.*reshape(mu, 1, 1, M).*eye(N);
end

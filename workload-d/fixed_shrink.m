function S = fixed_shrink(S, K, rho)
%FIXED_SHRINK  Shrinkage toward the scaled identity with a fixed weight rho.
%   S = fixed_shrink(S, K, rho) returns K*((1-rho)*S/K + rho*(trace(S/K)/N)*I)
%   (sum scale in, sum scale out; trace is preserved). S : N x N x M.

    N = size(S, 1); M = size(S, 3);
    Sv = reshape(S, N*N, M);
    mu = real(sum(Sv(1:N+1:end, :), 1)) / N;          % trace(S)/N per page (= K*mu of S/K)
    S = (1 - rho)*S + rho*reshape(mu, 1, 1, M).*eye(N);
    % K*(rho*trace(S/K)/N) = rho*trace(S)/N, so K drops out; kept in the signature
    % so every transform has the same (S, K, ...) form.
end

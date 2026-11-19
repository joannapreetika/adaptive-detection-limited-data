function T = detect(x, S, p)
%DETECT  AMF (Wald), Kelly GLRT and Rao statistics for M trials.
%   T = detect(x, S, p)
%   x : N x M      test vectors
%   S : N x N x M  covariance matrices, one per trial (unnormalised S = sum x_k x_k',
%                  or any transform of it, e.g. loaded S + K*lam*I)
%   p : N x 1      steering vector
%   T : M x 3      columns [AMF, Kelly, Rao]
%
%   With tA = |p'S^-1 x|^2 / (p'S^-1 p) and beta = x'S^-1 x:
%       AMF   = tA
%       Kelly = tA / (1 + beta)
%       Rao   = tA / ((1 + beta)(1 + beta - tA))
%   Rao is |p'(S+xx')^-1 x|^2 / (p'(S+xx')^-1 p), rewritten with Sherman-Morrison,
%   so it holds for any S passed in.

    [N, M] = size(x);
    B = [reshape(x, N, 1, M), repmat(p, 1, 1, M)];    % N x 2 x M right-hand sides
    Z = pagemldivide(S, B);                            % [S^-1 x, S^-1 p] per trial
    Six = reshape(Z(:, 1, :), N, M);
    Sip = reshape(Z(:, 2, :), N, M);

    num  = abs(p' * Six).^2;                           % |p'S^-1 x|^2
    den  = real(p' * Sip);                             % p'S^-1 p
    beta = real(sum(conj(x) .* Six, 1));               % x'S^-1 x

    tA = num ./ den;
    T = [tA; tA ./ (1 + beta); tA ./ ((1 + beta) .* (1 + beta - tA))].';
end

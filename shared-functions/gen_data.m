function [x, X] = gen_data(R, K, M, alpha, p)
%GEN_DATA  M independent trials of test vector + K training snapshots.
%   [x, X] = gen_data(R, K, M, alpha, p)
%   x : N x M      test vectors,  x = alpha*p + n,  n ~ CN(0,R)   (alpha = 0 -> H0)
%   X : N x K x M  training snapshots, each column ~ CN(0,R), target-free
%   alpha : complex amplitude (scalar). With SNR = |alpha|^2 p^H R^-1 p, use
%           alpha = sqrt(SNR/real(p'*(R\p))). Its phase does not affect any of the
%           detectors (the noise is circularly symmetric), so a real alpha is fine.
%   p     : steering vector (only needed when alpha ~= 0).
%
%   Memory: X holds N*K*M complex doubles (16 bytes each), e.g. N=16, K=48,
%   M=2e4 -> ~250 MB. For large M call this in chunks and accumulate results.
%   Random numbers come from the global stream: set rng() in the calling script.

    N = size(R, 1);
    L = chol(R, 'lower');                           % R = L*L'
    cn = @(varargin) (randn(varargin{:}) + 1j*randn(varargin{:}))/sqrt(2);   % CN(0,1)

    X = reshape(L*reshape(cn(N, K*M), N, K*M), N, K, M);
    x = L*cn(N, M);
    if nargin >= 4 && alpha ~= 0
        x = x + alpha*p;
    end
end

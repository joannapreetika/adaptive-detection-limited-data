function S = load_S(S, K, lnr_dB, sigma2)
%LOAD_S  Diagonal loading of a sum-scale sample covariance.
%   S = load_S(S, K, lnr_dB, sigma2) returns S + K*delta*I, delta = sigma2*10^(lnr_dB/10),
%   i.e. Rhat = S/K + delta*I kept at sum scale (as detect/calib expect).
%   S : N x N x M (any number of pages).

    if nargin < 4, sigma2 = 1; end
    N = size(S, 1);
    S = S + K*sigma2*10^(lnr_dB/10)*eye(N);
end

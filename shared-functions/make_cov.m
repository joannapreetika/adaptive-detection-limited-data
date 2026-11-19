function R = make_cov(N, sigma2, theta_j, jnr_dB)
%MAKE_COV  Model A interference covariance: white noise plus J jammers.
%   R = make_cov(N, sigma2, theta_j, jnr_dB) returns
%       R = sigma2*I + sum_j P_j a(theta_j) a(theta_j)^H,   P_j = sigma2*10^(jnr_dB(j)/10)
%   theta_j : jammer angles in DEGREES (empty = noise only)
%   jnr_dB  : per-element jammer-to-noise ratio(s); a scalar applies to all jammers.

    J = numel(theta_j);
    R = sigma2*eye(N);
    if J == 0, return; end
    if isscalar(jnr_dB), jnr_dB = repmat(jnr_dB, 1, J); end
    A = steer(N, theta_j);                          % N x J
    P = sigma2*10.^(jnr_dB(:)/10);                  % J x 1 jammer powers
    R = R + A*diag(P)*A';
    R = (R + R')/2;                                 % exact Hermitian symmetry
end

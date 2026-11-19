function p = steer(N, theta)
%STEER  ULA steering vector(s), half-wavelength spacing.
%   p = steer(N, theta) returns the N x numel(theta) matrix whose columns are
%       a(theta) = [1, e^{j*pi*sin(theta)}, ..., e^{j*(N-1)*pi*sin(theta)}].'
%   theta is in DEGREES (0 = broadside). Not normalised: steer(N,0) = ones(N,1).

    n = (0:N-1).';
    p = exp(1j*pi*n*sind(theta(:).'));
end

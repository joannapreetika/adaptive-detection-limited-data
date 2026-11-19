function SNR_dB = snr_clair(Pd, Pfa)
%SNR_CLAIR  SNR (dB) at which the clairvoyant (known-R) matched filter reaches Pd at Pfa.
%   SNR_dB = snr_clair(Pd, Pfa)
%   Known-R matched filter: T = |p'R^-1 x|^2 / (p'R^-1 p) is Exp(1) under H0, so
%   eta = ln(1/Pfa) and Pd = Q1(sqrt(2 SNR), sqrt(2 eta)) (Marcum Q), with SNR the
%   output SINR. Same computation as Workload C (c2_pd_vs_loading.m), so the operating
%   point matches theirs (10.8 dB for Pd = 0.9, Pfa = 1e-3).

    eta = log(1/Pfa);
    SNR = fzero(@(s) pd_mf(s, eta) - Pd, [1 100]);
    SNR_dB = 10*log10(SNR);
end

function pd = pd_mf(snr, eta)
% Noncentral-exponential tail (= Marcum Q1), using the scaled Bessel function.
    f = @(t) besseli(0, 2*sqrt(snr*t), 1) .* exp(2*sqrt(snr*t) - t - snr);
    pd = integral(f, eta, Inf);
end

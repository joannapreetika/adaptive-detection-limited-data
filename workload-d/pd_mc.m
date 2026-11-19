function rate = pd_mc(R, K, M, alpha, a_sig, p, Strans, eta, m_out, alpha_o)
%PD_MC  Monte Carlo detection (or false-alarm) rate of AMF, Kelly and Rao.
%   rate = pd_mc(R, K, M, alpha, a_sig, p, Strans, eta)
%   rate = pd_mc(R, K, M, alpha, a_sig, p, Strans, eta, m_out, alpha_o)
%   R       : interference covariance (data are drawn from it)
%   K, M    : training size, number of trials (run in chunks of 1e4)
%   alpha   : target amplitude in the cell under test (0 -> actual Pfa)
%   a_sig   : steering vector the target ARRIVES from (may differ from p: mismatch)
%   p       : steering vector the detector ASSUMES
%   Strans  : transform of S (handle @(S) or @(S, X)), or a cell of such handles;
%             [] = no transform. Every transform is applied to the SAME data.
%   eta     : thresholds, nT x 3 (one row per transform)
%   m_out   : number of contaminated training cells (default 0)
%   alpha_o : outlier amplitude; the first m_out training cells get
%             alpha_o*exp(j*phi)*p, phi ~ U[0, 2pi) independently per cell and trial
%   rate    : nT x 3 fraction of trials with T > eta, columns [AMF, Kelly, Rao]
%
%   Random numbers come from the global stream: set rng() before calling. With
%   m_out = 0 no extra numbers are drawn, so the clean case is reproduced exactly.

    if nargin < 7 || isempty(Strans), Strans = @(S) S; end
    if ~iscell(Strans), Strans = {Strans}; end
    if nargin < 9, m_out = 0; alpha_o = 0; end
    nT = numel(Strans);
    useX = cellfun(@(f) nargin(f) >= 2, Strans);

    chunk = 1e4; hits = zeros(nT, 3);
    for m0 = 1:chunk:M
        m = min(chunk, M - m0 + 1);
        [x, X] = gen_data(R, K, m, alpha, a_sig);
        if m_out > 0
            ph = exp(1j*2*pi*rand(1, m_out, m));
            X(:, 1:m_out, :) = X(:, 1:m_out, :) + alpha_o*p.*ph;
        end
        S0 = pagemtimes(X, 'none', X, 'ctranspose');
        for it = 1:nT
            if useX(it), S = Strans{it}(S0, X); else, S = Strans{it}(S0); end
            hits(it, :) = hits(it, :) + sum(detect(x, S, p) > eta(it, :), 1);
        end
    end
    rate = hits / M;
end

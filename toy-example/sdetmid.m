%% toy_glrt_wald_rao_modelA.m
% Wald (AMF), GLRT (Kelly) and Rao: toy example + radar-style Model A simulation.
%   Toy problem: H0: A=0  vs  H1: A>0 (sign KNOWN, one-sided, as in Kay Ex. 3.2).
%   Statistics are set to 0 when xbar<=0 (the MLE of A under H1 is max(xbar,0)).
%   The asymptotic null law is a half-chi2_1 mixture, so the asymptotic threshold
%   for PFA=alpha is the chi2_1 point at 2*alpha.
%   Part A : DC + white noise, sigma^2 unknown, NO training data
%   Part B : DC + white noise, sigma^2 estimated from K training vectors, +- loading

% Base MATLAB only (no toolbox): uses erfcinv / randg.
% NOTE: untested template; angles, powers, N, delta are placeholders to tune.
clear; clc; close all; rng(1);

Q    = @(x) 0.5*erfc(x/sqrt(2));
Qinv = @(p) sqrt(2)*erfcinv(2*p);
chi2thr = @(alpha) 2*erfinv(1-alpha).^2;      % chi^2_1 upper-alpha point
pfa_grid = logspace(-3,0,40);
col = struct('W',[0.85 0.33 0.10],'G',[0 0.45 0.74],'R',[0.47 0.67 0.19]);

%% ====================== PART A: no training data ========================
fprintf('=== PART A: sigma^2 unknown, no training ===\n');
alpha = 0.01;  thr = chi2thr(2*alpha);         % one-sided: half-chi2_1 mixture
Nlist = [3 5 10 20 50 100];  M = 5e5;
pfa_act = zeros(3,numel(Nlist));
for i = 1:numel(Nlist)
    N  = Nlist(i);
    X  = randn(M,N);                           % H0, sigma^2 = 1 (value irrelevant)
    xb = mean(X,2);  ss = sum((X-xb).^2,2);
    t  = N*xb.^2./ss;  ind = (xb>0);           % one-sided: statistic = 0 if xbar<=0
    TW = N*t.*ind;  TG = N*log(1+t).*ind;  TR = N*t./(1+t).*ind;
    pfa_act(:,i) = [mean(TW>thr); mean(TG>thr); mean(TR>thr)];
end
fprintf('Nominal PFA = %.3g (asymptotic chi2_1 threshold = %.3f)\n',alpha,thr);
disp(array2table(pfa_act.','VariableNames',{'Wald','GLRT','Rao'}, ...
     'RowNames',cellstr("N="+string(Nlist))));

% ordering check + ROCs with EXACT (empirical) thresholds, small N
N = 4; M = 5e5; ENRdB = 10; A = sqrt(10^(ENRdB/10)/N);
[T0,T1] = deal(struct());
for h = 0:1
    X = randn(M,N) + h*A;
    xb = mean(X,2); ss = sum((X-xb).^2,2); t = N*xb.^2./ss; ind = (xb>0);
    S = struct('W',N*t.*ind,'G',N*log(1+t).*ind,'R',N*t./(1+t).*ind);
    if h==0, T0 = S; else, T1 = S; end
end
fprintf('Ordering TR<=TG<=TW holds for all H1 trials: %d\n', ...
        all(T1.R<=T1.G+1e-12 & T1.G<=T1.W+1e-12));

figure('Name','Part A');
subplot(1,2,1);
loglog(Nlist,pfa_act(1,:),'-o',Nlist,pfa_act(2,:),'-s',Nlist,pfa_act(3,:),'-^', ...
       Nlist,alpha*ones(size(Nlist)),'k--','LineWidth',1.4); grid on;
xlabel('N'); ylabel('actual P_{FA}');
title(sprintf('Asymptotic threshold, nominal P_{FA}=%.2g',alpha));
legend('Wald','GLRT','Rao','nominal','Location','best');
subplot(1,2,2);
pd = @(a,b) pd_at(a,b,pfa_grid);
loglog(pfa_grid,pd(T0.W,T1.W),'-o',pfa_grid,pd(T0.G,T1.G),'-s', ...
       pfa_grid,pd(T0.R,T1.R),'-^',pfa_grid,Q(Qinv(pfa_grid)-sqrt(10^(ENRdB/10))),'k--','LineWidth',1.2);
grid on; xlabel('P_{FA}'); ylabel('P_D');
title(sprintf('Exact thresholds, N=%d, ENR=%d dB: ROCs coincide',N,ENRdB));
legend('Wald','GLRT','Rao','clairvoyant (\sigma^2 known)','Location','southeast');

%% ====================== PART B: with training data ======================
fprintf('\n=== PART B: sigma^2 estimated from K training vectors ===\n');
N = 4; M = 4e5; ENRdB = 10; A = sqrt(10^(ENRdB/10)/N);
Klist = [1 2 8 64];  delta = 0.3;           % sigma^2 = 1

figure('Name','Part B');
for ik = 1:numel(Klist)
    K = Klist(ik);
    c = 2*randg(K*N/2, M, 1);                % training energy ~ chi2_{KN}
    Z = randn(M,N);                          % common random numbers for H0/H1
    stat = @(h,dl) toy_stats(Z + h*A, c, K, N, dl);
    s0 = stat(0,0);   s1 = stat(1,0);
    d0 = stat(0,delta); d1 = stat(1,delta);
    subplot(2,2,ik); hold on;
    plot_roc(pfa_grid,s0.W,s1.W,col.W,'-'); plot_roc(pfa_grid,s0.G,s1.G,col.G,'-');
    plot_roc(pfa_grid,s0.R,s1.R,col.R,'-');
    plot_roc(pfa_grid,d0.W,d1.W,col.W,'--'); plot_roc(pfa_grid,d0.G,d1.G,col.G,'--');
    plot_roc(pfa_grid,d0.R,d1.R,col.R,'--');
    plot(pfa_grid,Q(Qinv(pfa_grid)-sqrt(10^(ENRdB/10))),'k:','LineWidth',1.2);
    set(gca,'XScale','log','YScale','log'); grid on;
    xlabel('P_{FA}'); ylabel('P_D'); title(sprintf('N=%d, K=%d, ENR=%d dB',N,K,ENRdB));
    if ik==1
        legend('Wald (AMF)','GLRT','Rao','Wald+DL','GLRT+DL','Rao+DL','clairvoyant','Location','southeast');
    end
    fprintf('K=%2d  PD@PFA=1e-2: W=%.3f G=%.3f R=%.3f | +DL: W=%.3f G=%.3f R=%.3f\n',K, ...
        pd_at(s0.W,s1.W,1e-2),pd_at(s0.G,s1.G,1e-2),pd_at(s0.R,s1.R,1e-2), ...
        pd_at(d0.W,d1.W,1e-2),pd_at(d0.G,d1.G,1e-2),pd_at(d0.R,d1.R,1e-2));
end




%% ============================ functions =================================
function [R,p,Rh,alphaAmp] = modelA(N,sig2,th_j,Pj_dB,th_t,ENRdB)
% Model A covariance on a half-wavelength ULA.
    a  = @(th) exp(1j*pi*(0:N-1).'*sin(th));     % ULA steering vector
    R  = sig2*eye(N);
    for j = 1:numel(th_j)
        aj = a(th_j(j));
        R  = R + 10^(Pj_dB(j)/10)*(aj*aj');
    end
    R  = (R+R')/2;
    p  = a(th_t);                                % target steering vector
    Rh = sqrtm(R);  Rh = (Rh+Rh')/2;             % Hermitian square root
    alphaAmp = sqrt(10^(ENRdB/10)/real(p'*(R\p)));   % ENR = |alpha|^2 p'R^-1 p
end

function [S,x0,x1] = draw_trial(Rh,p,alphaAmp,N,K,leak)
% One Monte Carlo trial: K target-free training vectors + one test vector.
    cn = @(r,c) (randn(r,c)+1j*randn(r,c))/sqrt(2);   % CN(0,I)
    Xt = Rh*cn(N,K);                                   % training: columns ~ CN(0,R)
    if leak > 0                                        % optional target leakage
        Xt(:,1) = Xt(:,1) + leak*alphaAmp*p;
    end
    S  = Xt*Xt';                                       % sum of outer products
    w  = Rh*cn(N,1);
    x0 = w;                                            % H0
    x1 = alphaAmp*p + w;                               % H1
end

function pd = pd_at(T0,T1,pfa)
% Monte Carlo ROC point: threshold = empirical (1-PFA) quantile of H0 stats
    if any(isnan(T0(:))) || any(isnan(T1(:))), pd = nan(size(pfa)); return; end
    T0 = sort(T0(:)); M0 = numel(T0); pd = zeros(size(pfa));
    for i = 1:numel(pfa)
        k = min(M0,max(1,ceil(M0*(1-pfa(i)))));
        pd(i) = mean(T1(:) > T0(k));
    end
end

function plot_roc(pfa,T0,T1,c,ls)
    plot(pfa,pd_at(T0,T1,pfa),'Color',c,'LineStyle',ls,'LineWidth',1.4);
end

function s = toy_stats(X,c,K,N,delta)
% Scalar-sigma^2 toy. a = N*xbar^2 (DC energy), b = residual energy, c = training energy.
    xb = mean(X,2);  a = N*xb.^2;  b = sum((X-xb).^2,2);
    ind = (xb>0);                         % one-sided test (A>0 known): 0 if xbar<=0
    sW = c/(K*N);                         % Wald/AMF: training data only
    s0 = (a+b+c)/((K+1)*N);               % H0-restricted MLE (uses x)  -> Rao
    s1 = (b+c)/((K+1)*N);                 % H1 MLE                       -> GLRT
    s.W = a./(sW+delta).*ind;
    s.R = a./(s0+delta).*ind;
    s.G = (K+1)*N*log((s0+delta)./(s1+delta)).*ind;
end

function t = clair(R,p,x)
% Clairvoyant (true R) whitened matched filter, complex data
    t = abs(p'*(R\x))^2/real(p'*(R\p));
end

function t = adapt_stats(S,K,x,p,delta)
% returns [AMF Kelly Rao] using R_hat = S/K + delta*I (Rao: (S+xx')/(K+1) + delta*I)
% Complex data: Hermitian transposes and |.|^2.  NaN if delta=0 and K<N (S singular).
    N = numel(x);
    if delta == 0 && K < N, t = nan(1,3); return; end
    I   = eye(N);
    Rh  = S/K + delta*I;   Rh0 = (S+x*x')/(K+1) + delta*I;
    ix  = Rh\x;  ip = Rh\p;
    num = abs(p'*ix)^2;  den = real(p'*ip);
    amf   = num/den;
    kelly = num/(den*(K + real(x'*ix)));   % Kelly GLRT, written with R_hat = S/K
    iz  = Rh0\x; jp = Rh0\p;
    rao = abs(p'*iz)^2/real(p'*jp);
    t = [amf kelly rao];
end
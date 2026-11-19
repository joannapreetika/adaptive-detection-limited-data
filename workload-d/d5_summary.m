%% D5: summary numbers for takeaways.md, read only from the D1-D4 results files
% Every number quoted in takeaways.md should appear in this script's output.
% Run after d1-d4. SDET_QUICK=1 summarises the quick runs in results/quick/ instead.

clear; clc;
quick = strcmp(getenv('SDET_QUICK'), '1');
outdir = 'results'; if quick, outdir = 'results/quick'; end
r1 = load(fullfile(outdir, 'd1_steering_mismatch.mat'));
r2 = load(fullfile(outdir, 'd2_contaminated_training.mat'));
r3 = load(fullfile(outdir, 'd3_contamination_loading.mat'));
r4 = load(fullfile(outdir, 'd4_shrinkage_vs_loading.mat'));
tests = r1.tests; scen = r1.scen;
fprintf('Results from %s/ (quick = %d)\n', outdir, quick);

%% D1: steering mismatch
fprintf('\n===== D1: steering mismatch =====\n');
fprintf('Operating SNR: %.1f dB (M0 = %g, M1 = %g, seed = %d)\n', r1.SNR_dB, r1.M0, r1.M1, r1.seed);
ik = find(r1.Klist == 32);
for is = 1:numel(scen)
    fprintf('%s, K = 32, dtheta = 0: Pd AMF %.3f, Kelly %.3f, Rao %.3f\n', scen{is}, ...
        squeeze(r1.Pd(is, ik, 1, :)));
end
for is = 1:numel(scen)
    fprintf('\n%s: dtheta (deg) where Pd < 0.5\n', scen{is});
    disp(array2table(round(squeeze(r1.d50(is, :, :)), 2), 'VariableNames', tests, ...
        'RowNames', compose('K=%d', r1.Klist)));
end
for d = [1 2 3 4]
    id = find(r1.dtheta == d);
    for is = 1:numel(scen)
        fprintf('%s, K = 32, dtheta = %g deg (cos^2 = %.3f): Pd AMF %.3f, Kelly %.3f, Rao %.3f\n', ...
            scen{is}, d, r1.cos2(is, id), squeeze(r1.Pd(is, ik, id, :)));
    end
end

%% D2: contaminated training
fprintf('\n===== D2: contaminated training (clean thresholds) =====\n');
ik = find(r2.Klist == 32);
[~, isnr] = min(abs(r2.SNR_cut - r1.SNR_dB));
for is = 1:numel(scen)
    fprintf('\n%s, K = 32: Pd at CUT SNR = %d dB | actual Pfa / 1e-3 | SNR loss at Pd = 0.9 (dB)\n', ...
        scen{is}, r2.SNR_cut(isnr));
    Tb = [squeeze(r2.Pd(is, ik, :, isnr, :)), squeeze(r2.Pfa_emp(is, ik, :, :))/r2.Pfa, ...
          squeeze(r2.loss90(is, ik, :, :))];
    disp(array2table(round(Tb, 3), 'RowNames', r2.case_name, 'VariableNames', ...
        [strcat('Pd_', tests), strcat('Pfa_', tests), strcat('loss_', tests)]));
end
for is = 1:numel(scen)
    fprintf('%s: SNR loss at Pd = 0.9, m = 1, outlier 20 dB, per K (AMF / Kelly / Rao):\n', scen{is});
    ic = find(r2.case_m == 1 & r2.case_SNRo == 20);
    for ik2 = 1:numel(r2.Klist)
        fprintf('  K = %d: %.2f / %.2f / %.2f dB\n', r2.Klist(ik2), squeeze(r2.loss90(is, ik2, ic, :)));
    end
end

%% D3: light loading under contamination
fprintf('\n===== D3: light loading under contamination (SNR = %.1f dB) =====\n', r3.SNR_dB);
ik = find(r3.Klist == 32);
for is = 1:numel(scen)
    fprintf(['\n%s, K = 32: Pd clean-unloaded | contaminated unloaded | best LNR (dB) -> Pd | ' ...
             'clean Pd at that LNR | Pfa/1e-3 there\n'], scen{is});
    for ic = 2:numel(r3.case_m)
        for it = 1:3
            lb = r3.LNR_best(is, ik, ic, it);
            il = find(r3.LNR_dB == lb) + 1;
            fprintf('  %-10s %-5s: %.3f | %.3f | %2d -> %.3f | %.3f | %.2f\n', r3.case_name{ic}, tests{it}, ...
                r3.Pd(is, ik, 1, 1, it), r3.Pd(is, ik, ic, 1, it), lb, r3.Pd_best(is, ik, ic, it), ...
                r3.Pd(is, ik, 1, il, it), r3.Pfa_emp(is, ik, ic, il, it)/r3.Pfa);
        end
    end
end

%% D4: shrinkage vs loading
fprintf('\n===== D4: shrinkage vs loading (SNR = %.1f dB) =====\n', r4.SNR_dB);
for is = 1:numel(scen)
    for it = 1:3
        fprintf('\n%s, %s:\n', scen{is}, tests{it});
        Tb = table(r4.Klist.', round(r4.Pd_plain(is, :, it).', 3), round(r4.Pd_bload(is, :, it).', 3), ...
            r4.LNR_best(is, :, it).', round(r4.Pd_bfs(is, :, it).', 3), r4.rho_best(is, :, it).', ...
            round(r4.rho_best_lnr(is, :, it).', 1), ...
            round(r4.Pd_lw(is, :, it).', 3), round(r4.gap_lw(is, :, it).', 3), ...
            round(r4.rho_lw_mean(is, :).', 3), ...
            'VariableNames', {'K', 'plain', 'best_load', 'LNR', 'best_fs', 'rho', 'rho_as_LNR', ...
                              'LW', 'LW_gap', 'mean_rho_LW'});
        disp(Tb);
    end
end
fprintf('rho -> equivalent loading on true R (dB re sigma2):\n');
for is = 1:numel(scen)
    fprintf('  %s: rho = %s\n       LNR = %s\n', scen{is}, mat2str(r4.rho_fs, 2), ...
        mat2str(round(r4.lnr_equiv(is, :), 1)));
end
fprintf('Largest LW shortfall vs tuned optimum: ');
[g, idx] = min(r4.gap_lw(:));
[is, ik, it] = ind2sub(size(r4.gap_lw), idx);
fprintf('%.3f (%s, %s, K = %d)\n', g, scen{is}, tests{it}, r4.Klist(ik));

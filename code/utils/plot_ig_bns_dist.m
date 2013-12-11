%% PLOT_IG_BNS_DIST - Distribution of IG and BNS

startup;

% uncomment to calculate these!
IG = calc_information_gain(Y,X,1:size(X,2),0);
bns = calc_bns(Y,X);

% number of points
N = 100;

ig_vals = linspace(min(IG),max(IG),N);
bns_vals = linspace(min(bns),max(bns),N);

counts_ig = zeros(N,1);
counts_bns = zeros(N,1);

for i=1:N
   
    count_ig = nnz(IG >= ig_vals(i));
    count_bns = nnz(bns >= bns_vals(i));
    
    counts_ig(i) = count_ig;
    counts_bns(i) = count_bns;
    
end

figure;
subplot(2, 1, 1);
semilogy(ig_vals / max(ig_vals), counts_ig, 'b');
hold on;
semilogy(bns_vals / max(bns_vals), counts_bns, 'r');
title('Compared: IG and BNS');
ylabel('# of features retained');
xlabel('Threshold (Normalized)');
legend('IG', 'BNS');

subplot(2, 1, 2);
loglog(counts_ig, ig_vals, 'b');
hold on;
loglog(counts_bns, bns_vals, 'r');
title('Compared: IG and BNS');
xlabel('# of features retained');
ylabel('Threshold (Non-normalized)');
legend('IG', 'BNS');




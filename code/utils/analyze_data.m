addpath('~/Workspace/matlab/')
close all
[N, M] = size(X);
% how many words in each review
figure()
plot(1:N, sort(sum(X,2)))
grid on
change_line(gcf, 2)

% how many times a word has been used
figure()
plot(1:M, sort(sum(X,1)))
grid on
change_line(gcf, 2)

freq_ind = (sum(X,1) > 5000);
freq_word = vocab(freq_ind);

low_freq_ind = (sum(X,1) < 3);
low_freq_word = vocab(low_freq_ind);

word_count = sort(sum(X,1), 'descend');
high_freq_ind = (sum(X,1) >= word_count(3));
high_freq_word = vocab(high_freq_ind);

bns_thresh = 0.01;
[bns, bns_ind] = calc_bns(X, Y, bns_thresh);

fprintf('bns feature: \t\t%d.\n', nnz(bns_ind))
fprintf('low_freq feature: \t%d.\n', nnz(low_freq_ind))
fprintf('intersection low: \t%d.\n', nnz(bns_ind & low_freq_ind))
fprintf('high_freq feature: \t%d.\n', nnz(high_freq_ind))
fprintf('intersection high: \t%d.\n', nnz(bns_ind & high_freq_ind))
%% GEN_TOP_WORDS - Generate list of top words for both BNS and IG

startup;

N = 500;

IG = fastig(X,Y,1:size(X,2),[1 2 3 4 5]);
bns = calc_bns(X,Y);

[~, top_ig] = sort(IG, 'descend');
[~, top_bns] = sort(bns, 'descend');

voc_ig = vocab(top_ig(1:N));
voc_bns = vocab(top_bns(1:N));

f = fopen('vocab.txt', 'w');
for i=1:N
    fprintf(f, '%s\t\t\t%s\n', voc_bns{i}, voc_ig{i});
end
fclose(f);

fprintf('Of the %i top words, IG and BNS have %i in common\n', N, numel(intersect(top_ig(1:N),top_bns(1:N))));

for n=1:1000:numel(vocab)
   count(n) = numel(intersect(top_ig(1:n),top_bns(1:n)));
end

plot(count);

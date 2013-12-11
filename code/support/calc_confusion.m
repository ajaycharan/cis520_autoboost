function [ mtx ] = calc_confusion( Ypred, Ytest )
%CALC_CONFUSION Calculate the confusion matrix

% get all unique labels
labels = unique(Ytest);
M = numel(labels);

mtx = zeros(M,M);

for i=1:M
    for j=1:M
        label_is_j = (Ytest == labels(j));
        
        % column j = should predict label j
        % row i = predicted label i
        
        mtx(i,j) = nnz(Ypred(label_is_j) == labels(i)) / nnz(label_is_j);
                
    end
end
end


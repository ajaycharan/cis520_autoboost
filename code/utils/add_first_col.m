function Kout = add_first_col(K)
% To use precomputed kernel, you must include sample serial number as
% the first column of the training and testing data (assume your kernel
% matrix is K, # of instances is n):
Kout = [(1:size(K, 1))', K];
end
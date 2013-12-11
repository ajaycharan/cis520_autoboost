function [ model, info ] = train_liblinear( Xtrain, Ytrain, s, c )
% [ model, info ] = train_liblinear( Xtrain, Ytrain, s, c )
% Trains a SVM using liblinear and evaluates on test data.
% type : set type of solver (default 1)
%   for multi-class classification
% 	 1 -- L2-regularized L2-loss support vector classification (dual)
% 	 3 -- L2-regularized L1-loss support vector classification (dual)
% 	 5 -- L1-regularized L2-loss support vector classification
%    6 -- L1-regularized logistic regression
% 	 7 -- L2-regularized logistic regression (dual)
% 	12 -- L2-regularized L2-loss support vector regression (dual)
% 	13 -- L2-regularized L1-loss support vector regression (dual)
% Xtrain - train data
% Ytrain - train label
% s      - type for -s option
% c      - cost parameter

% Default option
if nargin < 3, s = 1; end
if nargin < 4, c = 0.1; end

% Display type
type_num = [0 1 2 3 4 5 6 7 12 13];
type_ind = (type_num == s);
if ~nnz(type_ind), error('Wrong model type: %d.', s); end
% type_name = {...
%     'L2-regularized L2-loss SVM (dual)', ...
%     'L2-regularized L1-loss SVM (dual)', ...
%     'L1-regularized L2-loss SVM', ...
%     'L1-regularized LR', ...
%     'L2-regularized LR (dual)', ...
%     'L2-regularized L2-loss SVR (dual)',...
%     'L2-regularized L1-loss SVR (dual)'};

% Train model
tic
% fprintf('***** Strat training %s with C = %g. \n', type_name{type_ind}, c)
option  = sprintf('-s %d -q -c %g', s, c);

model   = liblinear_train(Ytrain, Xtrain, option);
time = toc;
% fprintf('***** Finish training %s with time: %f. \n', type_name{type_ind}, time)

% Save model info
info.option = option;
info.type   = s;
info.cost   = c;
info.time   = time;
end

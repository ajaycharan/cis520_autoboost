%% TUNE_ENSEMBLE.m - Load and run all model, generate ensemble.mat
%
% Copyright (c) 2013, Chao Qu, Gareth Cross, Pimkhuan Hannanta-Anan. All
% rights reserved.
%   Contact: quchao@seas.upenn.edu, gcross@seas.upenn.edu, pimh@seas.upenn.edu
%
% Re-loads the environment, then loads all save model files. Model
% parameters are used to re-train all models and generate validation
% scores. Finally, training predictions are used to calculate ensemble
% coefficeints using ridge regression. Coefficients are saved to
% ensemble.mat.
%
% Description of options available for this script:
%
%   LIBLINEAR_MODEL_FILES
%       List of model files generated using tune_liblinear, without bagging.
%   NB_MODEL_FILES
%       List of model files generated using tune_nb, without bagging.
%   BAGGED_LIBLINEAR_MODEL_FILES
%       List of model files generated using tune_liblinear, with bagging.
%   BAGGED_NB_MODEL_FILES
%       List of model files generated using tune_nb, with bagging.
%
% Note: The performance of bagged models significantly exceeds that of
% unbagged models. For this reason, unbagged models do not appear in the
% final solution and are loaded _only_ in this file.

startup;

% Options + Models

% Models generated using liblinear
LIBLINEAR_MODEL_FILES = {};

% Models generated using Naive Bayes
NB_MODEL_FILES = {};

% Models generated using bagged liblinear
BAGGED_LIBLINEAR_MODEL_FILES = {'l2rl2ld_svm_bag', 'l2rd_lr_bag', 'l2rl1ld_svm_bag', ...
     'l1r_lr_bag', 'l2rl2ld_svm_bi_bag', 'l1r_lr_bi_bag', 'l2rl1ld_svm_bi_bag'};
 
% Models generated using bagged bayes
BAGGED_NB_MODEL_FILES = {'nb_mn_bag', 'nb_mn_bi_bag'};

% use all data for BNS scores
bns = calc_bns(X, Y);

Ytrain_hat = [];    % training predictions for all models
Yvalid_hat = [];    % validation predictions for all models

valid_rmse = [];    % RMSE of all individual models, in order of execution

tic;

%% Process liblinear models
for n=1:numel(LIBLINEAR_MODEL_FILES)
    
    linear = load(LIBLINEAR_MODEL_FILES{n});
    
    feat_ind = linear.info.feat_ind;
    c = linear.info.cost;               % cost parameter
    s = linear.info.type;               % type of regression
    
    % apply BNS scaling + threshold
    Xtrain_n = double(Xtrain > 0);
    Xvalid_n = double(Xvalid > 0);
    
    Xtrain_n = bsxfun(@times, Xtrain_n, bns);
    Xtrain_n = Xtrain_n(:, feat_ind);
    Xvalid_n = bsxfun(@times, Xvalid_n, bns);
    Xvalid_n = Xvalid_n(:, feat_ind);
    
    model = train_liblinear(Xtrain_n, Ytrain, s, c);
    
    % predict both training and validation
    Ytrain_n = predict_liblinear(model, Xtrain_n);
    Yvalid_n = predict_liblinear(model, Xvalid_n);
    
    Ytrain_hat = [Ytrain_hat Ytrain_n];
    Yvalid_hat = [Yvalid_hat Yvalid_n];
    
    valid_rmse(end+1) = rms(Yvalid_n - Yvalid);
    
    fprintf('-- Validation RMSE for %s -> %f --\n', LIBLINEAR_MODEL_FILES{n}, valid_rmse(end));
end

%% Process NB models
for n=1:numel(NB_MODEL_FILES)

    bayes = load(NB_MODEL_FILES{n});
    feat_ind = bayes.info.feat_ind;
    
    Xtrain_n = Xtrain(:, feat_ind);
    model = train_nb(Xtrain_n, Ytrain);
    
    % training and validation
    Ytrain_n = predict_nb(model, Xtrain_n);
    Yvalid_n = predict_nb(model, Xvalid(:, feat_ind));
    
    Ytrain_hat = [Ytrain_hat Ytrain_n];
    Yvalid_hat = [Yvalid_hat Yvalid_n];
    
    valid_rmse(end+1) = rms(Yvalid_n - Yvalid);
    
    fprintf('-- Validation RMSE for %s -> %f --\n', NB_MODEL_FILES{n}, valid_rmse(end));
    
end

%% Process bagged liblinear models
for n=1:numel(BAGGED_LIBLINEAR_MODEL_FILES)
   
    bagged = load(BAGGED_LIBLINEAR_MODEL_FILES{n});
    bagged = bagged.model; % pull from loaded struct
    
    if isfield(bagged, 'bigram_fs_thresh')
        XT = Xd_train;
        XV = sparse(size(Xd_valid,1), size_train);
        XV(:, indices_train_in_valid) = Xd_valid(:, indices_valid_in_train);
        
        XV = FS_bigram_train.apply(XV);
    else
        XT = Xtrain;
        XV = Xvalid;
    end
   
    model = train_bagged_liblinear(XT, Ytrain, Mtrain,...
                                    'T', bagged.T, 'N', bagged.N,...
                                    'cost', bagged.cost,...
                                    'type', bagged.type,...
                                    'thresh_bns', bagged.thresh_bns,...
                                    'mode', bagged.mode,...
                                    'binary', bagged.binary);
                                
    Ytrain_n = predict_bagged_liblinear(model, XT, Mtrain);
    Yvalid_n = predict_bagged_liblinear(model, XV, Mvalid);
    
    Ytrain_hat = [Ytrain_hat Ytrain_n];
    Yvalid_hat = [Yvalid_hat Yvalid_n];
    
    valid_rmse(end+1) = rms(Yvalid_n - Yvalid);
    
    fprintf('-- Validation RMSE for %s -> %f --\n', BAGGED_LIBLINEAR_MODEL_FILES{n}, valid_rmse(end));
    
end

%% Process bagged NB models
for n=1:numel(BAGGED_NB_MODEL_FILES)
   
    bagged = load(BAGGED_NB_MODEL_FILES{n});
    bagged = bagged.model; % pull from loaded struct
    
    if isfield(bagged, 'bigram_fs_thresh')
        XT = Xd_train;
        XV = sparse(size(Xd_valid,1), size_train);
        XV(:, indices_train_in_valid) = Xd_valid(:, indices_valid_in_train);
        
        XV = FS_bigram_train.apply(XV);
    else
        XT = Xtrain;
        XV = Xvalid;
    end
    
    model = train_bagged_nb(XT, Ytrain, Mtrain,...
                            'T', bagged.T, 'N', bagged.N,...
                            'thresh_bns', bagged.thresh_bns,...
                            'mode', bagged.mode,...
                            'binary', bagged.binary);
                                
    Ytrain_n = predict_bagged_nb(model, XT, Mtrain);
    Yvalid_n = predict_bagged_nb(model, XV, Mvalid);
    
    Ytrain_hat = [Ytrain_hat Ytrain_n];
    Yvalid_hat = [Yvalid_hat Yvalid_n];
    
    valid_rmse(end+1) = rms(Yvalid_n - Yvalid);
    
    fprintf('-- Validation RMSE for %s -> %f --\n', BAGGED_NB_MODEL_FILES{n}, valid_rmse(end));
    
end

%% Ensemble
ensemble_rmse = rms(Yvalid - mean(Yvalid_hat,2));
weights = 1./valid_rmse;
weights = weights/sum(weights);
weighted_rmse = rms(Yvalid - sum(bsxfun(@times, Yvalid_hat, weights),2));
w = inv(Ytrain_hat' * Ytrain_hat + 55000*eye(size(Ytrain_hat, 2))) * Ytrain_hat' * Ytrain;
regression_rmse = rms(Yvalid - Yvalid_hat * w);

fprintf('\n=========================================');
fprintf('\n ** Average Ensemble RMSE: %.4f **\n', ensemble_rmse)
fprintf('=========================================\n');
fprintf('\n=========================================');
fprintf('\n ** Weighted Ensemble RMSE: %.4f **\n', weighted_rmse)
fprintf('=========================================\n');
fprintf('\n=========================================');
fprintf('\n ** Regression Ensemble RMSE: %.4f **\n', regression_rmse)
fprintf('=========================================\n');

fprintf('\nFinished running in %f seconds.\n', toc);
save('./model/ensemble.mat', 'w');
fprintf('Saved ensemble coefficients to ensemble.mat\n');

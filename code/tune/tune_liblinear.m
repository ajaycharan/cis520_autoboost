%% TUNE_LIBLINERAR -- tune all liblinear models
% tune_liblinear tunes all types of model provided by liblinear and 
%   save thme as mat files in model folder.
% bns_thresh    - BNS threshhold for feature selection, default 0.01
% xval          - levels of crossvalidation to pick cost for liblinear
%    0 -- no cross validation, try to load cost from existing model
%    1 -- level 1 cross validation, from 10^-3 to 10^2
%    2 -- level 2 cross validation
% model_type    - option s in liblinear_train
%    0 -- L2-regularized LR (primal)
% 	 1 -- L2-regularized L2-loss SVM (dual) *
%    2 -- L2-regularized L2-loss SVM (primal)
% 	 3 -- L2-regularized L1-loss SVM (dual)
%    4 -- support vector classification by Crammer and Singer
% 	 5 -- L1-regularized L2-loss SVM        *
%    6 -- L1-regularized LR
% 	 7 -- L2-regularized LR (dual)          *
% 	12 -- L2-regularized L2-loss SVR (dual)
% 	13 -- L2-regularized L1-loss SVR (dual)
% use_bigrams   - true for using bigrams, false for using unigrams
% use_binary    - true for using presence of word instead of counts
% use_bagging   - true for using bagged liblinear models, default true
% num_bags      - number of bags used in bagged models, default 10
% bag_mode      - 'average', 'vote', default average
% save_model    - true for saving models to model folder, default true
startup

bns_thresh  = 0.01;
xval        = 1;        % cross validation 1 - 1 level, 2 - 2 level
model_type  = 1;        % liblinear model type 1, 5, 7
type_all    = [0 1 2 3 4 5 6 7 12 13];
type_ind    = (type_all == model_type);
type_name   = {'l2rp_lr', 'l2rl2ld_svm', 'l2rl2lp_svm', 'l2rl1ld_svm', ...
               'cs_svm', 'l1rl2l_svm', 'l1r_lr', 'l2rd_lr', 'l2rl2ld_svr', 'l2rl1ld_svr'};
method      = type_name{type_ind};

% BEST TYPES: 1, 3, 5, 6, 7
% TYPES USED FOR BIGRAMS: 1, 3 and 6

use_bigrams = false;
save_model  = false;

use_binary  = false;     
use_bagging = true;      % bagging options
num_bags = 10;           % number of bags
bag_mode = 'average';    % 'average' or 'vote'

print_msg(method, 'start');

%% Feature selection
if ~use_bigrams
    Xtrain_set = Xtrain;
    Xvalid_set = Xvalid;
    Ytrain_set = Ytrain;
else
    % use_bigrams = true
    if ~exist('Xd_train', 'var') || ~exist('Xd_valid', 'var')
        error('!! Set BUILD_BIGRAMS=true in order to use bigrams !!');
    end

    % bigrams pre-thresholded in startup.m
    Xtrain_set = Xd_train;
    Ytrain_set = Ytrain;

    Xvalid_set = sparse(size(Xd_valid,1), size_train);
    Xvalid_set(:,indices_train_in_valid) = Xd_valid(:, indices_valid_in_train);

    Xvalid_set = FS_bigram_train.apply(Xvalid_set);

    fprintf('Bigrams: Pre-thresholded initial feature space to %i columns\n', size(Xtrain_set,2));
end

% generate file path
file_suffix = '';
if use_bigrams
    file_suffix = strcat(file_suffix, '_bi');
end
if use_bagging
    file_suffix = strcat(file_suffix, '_bag');
end
model_file  = sprintf('./model/%s%s.mat', method, file_suffix);

% do feature selection here for non-bagged models
if ~use_bagging
    FS = FeatureSelector.train(Xtrain_set, Ytrain_set, ...
                                'thresh_bns', bns_thresh,...
                                'binary', use_binary,...
                                'scale', 'bns');
    Xtrain_set = FS.apply(Xtrain_set);
    Xvalid_set = FS.apply(Xvalid_set);

    fprintf('%d features used with bns score %1.3f.\n', size(Xtrain_set, 2), bns_thresh);
end

%% Two level cross validation
for v = 1:xval
    if v == 1
        crange = 10.^(-3:2);
    elseif v == 2
        crange = logspace(log10(c)-0.9, log10(c)+0.9, 10);
    end
    xval_rmse = zeros(size(crange));
    for i = 1:numel(crange)
        if ~use_bagging
            train_fun = @(Xtr, Ytr) train_liblinear(Xtr, Ytr, model_type, crange(i));
            predict_fun  = @predict_liblinear;
        else
            % note: this form does not include passing metadata directly
            train_fun = @(Xtr, Ytr) train_bagged_liblinear(Xtr,Ytr,{},...
                                                'thresh_bns', bns_thresh,...
                                                'binary', use_binary,...
                                                'type', model_type,...
                                                'cost', crange(i),...
                                                'mode', bag_mode,...
                                                'T', num_bags);

            predict_fun = @(mdl, Xtest) predict_bagged_liblinear(mdl,Xtest,{});
        end

        xval_rmse(i) = cross_validate(Xtrain_set, Ytrain_set, 5, train_fun, predict_fun);
        fprintf('=== Cross Validation RMSE = %f. (c=%f)\n', xval_rmse(i),crange(i))
    end
    [best_rmse, best_ind] = min(xval_rmse);
    c = crange(best_ind);
    fprintf('Level %d cross-validation pick best c = %1.3f. \n', v, c)
end

if xval == 0
    try
        load(model_file);
        if ~use_bagging
            c = info.cost;
        else
            c = model.cost;
        end
        fprintf('No cross-val, using saved c = %1.3f instead\n', c);
    catch
        c = 3.162;
        fprintf('No cross-val, no saved model, using default c = %1.3f\n', c);
    end
end

%% Train model on Xtrain and test on Xvalid
if ~use_bagging
    model   = train_liblinear(Xtrain_set, Ytrain_set, model_type, c);
    Yhat    = predict_liblinear(model, Xvalid_set);
else

    model = train_bagged_liblinear(Xtrain_set,Ytrain_set,{}, 'thresh_bns', bns_thresh,...
                                                 'binary', use_binary,...
                                                 'type', model_type,...
                                                 'cost', c,...
                                                 'mode', bag_mode,...
                                                 'T', num_bags);

    Yhat = predict_bagged_liblinear(model,Xvalid_set,{});

end

valid_rmse = rms(Yhat - Yvalid);
fprintf('@@@ Final validation RMSE: %f. \n', valid_rmse)

%% Train model on X and save as mat file
if save_model
    if ~use_bagging
        X_set = X;
        if use_binary
            X_set = double(X_set > 0);
        end
        X_set  = bsxfun(@times, X_set, bns);
        X_set  = X_set(:, feat_ind);
        [model, info] = train_liblinear(X_set, Y, model_type, c);
        info.rmse = valid_rmse;
        info.name = method;
        info.feat_ind = feat_ind;
        print_msg(method, 'finsh');
        info.bns_thresh = bns_thresh;
        save(model_file, 'model', 'info');
    else
        % retrain on all data (NOTE: No metadata used!!)

        if use_bigrams
            % this is thresholded in startup.m
            XT = Xd;
        else
            XT = X;
        end

        model = train_bagged_liblinear(XT,Y,{},'thresh_bns', bns_thresh,...
                                               'binary', use_binary,...
                                               'type', model_type,...
                                               'cost', c,...
                                               'mode', bag_mode,...
                                               'T', num_bags);

        model.rmse = valid_rmse;    % append rmse to model
        if use_bigrams
            % append pre-selector to model
            model.bigram_fs_thresh = BIGRAM_PRE_THRESH;
        end

        save(model_file, 'model');
    end

    fprintf('Model %s saved to %s. \n', method, model_file);
end

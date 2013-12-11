startup
%% Model option
%startup
kernel      = 2; % intersection kernel
bns_thresh  = 0.01;
xval        = 0;        % cross validation 1 - 1 level, 2 - 2 level
method      = 'kernel_svm';
model_file  = sprintf('./model/%s.mat', method);
use_bigrams = false;
save_file   = false;
small_data  = 10000;
use_binary  = true;

print_msg(method, 'start');
%% Feature selection
if ~use_bigrams
    [bns, feat_ind] = calc_bns(Xtrain, Ytrain, bns_thresh);
%     Xtrain_set  = bsxfun(@times, Xtrain, bns);
%     Xvalid_set  = bsxfun(@times, Xvalid, bns);
    Xtrain_set = Xtrain;
    Xvalid_set = Xvalid;
    if use_binary
        Xtrain_set = double(Xtrain_set > 0);
        Xvalid_set = double(Xvalid_set > 0);
    end
    Xtrain_set  = bsxfun(@times, Xtrain_set, bns);
    Xvalid_set  = bsxfun(@times, Xvalid_set, bns);
    Xtrain_set  = Xtrain_set(:, feat_ind);
    Xvalid_set  = Xvalid_set(:, feat_ind);
else
    % use_bigras = true
    [bns, feat_ind] = calc_bns(Xd_train, Ytrain, bns_thresh);
    Xtrain_set  = bsxfun(@times, Xd_train, bns);
    
    Xvalid_set = sparse(size(Xd_valid,1), size(Xtrain_set,2));
    Xvalid_set(:,indices_train_in_valid) = Xd_valid(:, indices_valid_in_train);
    
    bsxfun(@times, Xvalid_set, bns);
    
    Xvalid_set  = Xvalid_set(:, feat_ind);
    Xtrain_set  = Xtrain_set(:, feat_ind);
end

% Use small dataset
if small_data > 0
    Xtrain_set = Xtrain_set(1:small_data, :);
    Ytrain = Ytrain(1:small_data);
end

fprintf('%d features used with bns score %1.3f. \n', nnz(feat_ind), bns_thresh)

%% Two level cross validation
for v = 1:xval
    if v == 1
        crange = 10.^(-3:3);
    elseif v == 2
        crange = linspace(c/5, c*5, 6);
    end
    xval_rmse = zeros(size(crange));
    for i = 1:numel(crange)
        train_fun    = @(Xtr, Ytr) train_libsvm(Xtr, Ytr, kernel, crange(i));
        predict_fun  = @(mdl, Xts, Xtr) predict_libsvm(mdl, Xts, Xtr, kernel);
        xval_rmse(i) = cross_validate(Xtrain_set, Ytrain, 5, train_fun, predict_fun);
        fprintf('=== Cross Validation RMSE = %f. (c=%f)\n', xval_rmse(i),crange(i))
    end
    [best_rmse, best_ind] = min(xval_rmse);
    c = crange(best_ind);
    fprintf('Level %d cross-validation pick best c = %1.3f. \n', v, c)
end

if xval == 0
    try
        load(model_file);
        c = info.c;
    catch
        c = 0.3;
    end
    fprintf('No cross-validation, use saved c = %1.3f instead. \n', c)
end

%% Train model on Xtrain and test on Xvalid
tic
model = train_libsvm(Xtrain_set, Ytrain, kernel, c);
Yhat = predict_libsvm(model, Xvalid_set, Xtrain_set, kernel);
Yhat(Yhat > 5) = 5;
Yhat(Yhat < 1) = 1;
valid_rmse = rms(Yhat - Yvalid);
fprintf('@@@ Final validation RMSE: %f. Time: %f \n', valid_rmse, toc)

%% Train model on X and save as mat file
% X_set  = bsxfun(@times, X, bns);
% X_set  = X_set(:, feat_ind);
% [model, info] = train_libsvm(X_set, Y, kernel, c);
% info.rmse = valid_rmse;
% info.name = method;
% info.feat_ind = feat_ind;
% print_msg(method, 'finsh');
% info.bns_thresh = bns_thresh;
% 
% if save_model
%     save(model_file, 'model', 'info');
% end

%% TUNE_NB.m - Tune NB using cross validation to select BNS threshold
%
% Copyright (c) 2013, Chao Qu, Gareth Cross, Pimkhuan Hannanta-Anan. All
% rights reserved.
%   Contact: quchao@seas.upenn.edu, gcross@seas.upenn.edu, pimh@seas.upenn.edu
%
% Tunes a multinomial Naive Bayes model in order to select the best
% possible BNS threshold for feature selection. 5-fold cross-validation is
% used for every BNS level tested.
%
% Description of options available for this script:
%
%   xval
%       Either 0, 1 or 2. Number of levels of search to perform for a
%       strong bns score. If 0 is selected, the value 'best_bns' must be
%       defined (some suggested values are available below). The script
%       only searches up 0.5*max(bns), where max(bns) is the largest bns
%       score in the entire dataset. Recommended: 2
%   file_prefix
%       Prefix to use when saving model file. Recommended: nb_mn
%   save_model
%       Set to true in order to train a final model on all data and save to
%       disk.
%   use_bagging
%       Set to true in order to use bagging. This will increase the
%       accuracy of the model without significantly impacting variance.
%   use_binary
%       Set to true in order to use binary features. Words will be counted
%       by presence in an observation, and not by frequency.
%   use_bigrams
%       Set to true to train on the bigram dataset. Note that BUILD_BIGRAMS
%       must be true in startup.m in order for this to work.
%   num_bags
%       Number of bags to use when training bagged models. Values above 10
%       do not seem to improve performance noticably.
%   bag_mode
%       Bagging mode to use. See train_bagged_nb.m, Recommended: 'average'

startup;                 % creates data

% Options
xval = 0;                % 1 or 2
file_prefix = 'nb_mn';   % prefix for model file
save_model = false;
use_bagging = true;
use_binary = true;
use_bigrams = true;
num_bags = 10;
bag_mode = 'average';

% Some recommended values:
%   best_bns = 0.192482; % for regular
%   best_bns = 0.157128; % for bigrams

if ~any(xval == [0 1 2])
    error('Invalid value selected for xval!');
end

if ~use_bigrams
    
    Xtrain_set = Xtrain;     % data to train/validate on
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

% find range of bns on all data
bns_all = calc_bns(X,Y);

for v=1:xval
   if v == 1
        bns_thresh = linspace(0.01, max(bns_all)*0.5, 15);
        %bns_thresh = [0.01 0.03 0.05 0.07 0.09 0.11 0.13 0.15 0.17 0.19 0.21];
    elseif v == 2
        range = max(bns_all) - min(bns_all);
        bns_thresh = linspace(best_bns - range*0.05, best_bns + range*0.05, 10);
   end
   
   rmse = zeros(numel(bns_thresh), 1);
   for i=1:numel(bns_thresh)
       
       if ~use_bagging
           % do feature selection
           FS = FeatureSelector.train(Xtrain_set, Ytrain_set, ...
                                    'thresh_bns', bns_thresh(i),...
                                    'binary', use_binary);
           
           Xt = FS.apply(Xtrain_set);
           
           %xval
           rmse(i) = cross_validate(Xt, Ytrain, 5, @train_nb, @predict_nb);
       else
           
           train_method = @(Xtr,Ytr)train_bagged_nb(Xtr,Ytr,{},...
                                    'T', num_bags,...
                                    'thresh_bns', bns_thresh(i),...
                                    'binary', use_binary,...
                                    'mode', bag_mode);
                             
           pred_method = @(mdl,Xtest)predict_bagged_nb(mdl,Xtest,{});
           
           rmse(i) = cross_validate(Xtrain_set, Ytrain, 5, train_method, pred_method);
       end
       
       fprintf('-- BNS=%f, RMSE=%f --\n', bns_thresh(i), rmse(i));
   end
   
   [best_rmse, best_idx] = min(rmse);
   best_bns = bns_thresh(best_idx);
   fprintf('\n\tBest BNS for round %i = %f (~%i feat)\n\n', v, best_bns, nnz(bns_all >= best_bns));
end

if xval==0 && ~exist('best_bns','var')
    error('Best bns must be specified in workspace if no xval selected');
end

fprintf('\nTraining validation model with BNS %f\n', best_bns);

if ~use_bagging
    FS = FeatureSelector.train(Xtrain_set, Ytrain_set, ...
                               'thresh_bns', best_bns,...
                               'binary', use_binary);
           
    Xt = FS.apply(Xtrain_set);
    Xv = FS.apply(Xvalid_set);
    
    model = train_nb(Xt, Ytrain_set);
    Ypred = predict_nb(model, Xv);
else
    model = train_bagged_nb(Xtrain_set,Ytrain_set,{},...
                            'T', num_bags,...
                            'thresh_bns', best_bns,...
                            'binary', use_binary,...
                            'mode', bag_mode);
                        
    Ypred = predict_bagged_nb(model, Xvalid_set);
end

fprintf('\n** Final validation RMSE: %f **\n', rms(Ypred - Yvalid));

if save_model
    file_suffix = '';
    if use_bigrams
        file_suffix = sprintf('%s%s', file_suffix, '_bi');
    end
    if use_bagging
        file_suffix = sprintf('%s%s', file_suffix, '_bag');
    end
    file_name = sprintf('./model/%s%s.mat', file_prefix, file_suffix);
    
    if use_bigrams
        Xfull = Xd;
    else
        Xfull = X;
    end
    
    if ~use_bagging
        error('Unbagged NB is deprecated! Don''t train this type of model!');
        
        info.rmse = rms(Ypred - Yvalid);
        info.feat_ind = features;
        info.bns_thresh = best_bns;
        info.use_bigrams = use_bigrams;
        
        % re-train on everything
        model = train_nb(X,Y);
        
        FS = FeatureSelector.train(Xfull, Y, ...
                                    'thresh_bns', best_bns,...
                                    'binary', use_binary);
           
        Xt = FS.apply(X);
    
        model = train_nb(Xt, Y);
        
        save(file_name, 'model', 'info');
    else
       
        model = train_bagged_nb(Xfull,Y,{},...
                                 'T', num_bags,...
                                 'thresh_bns', best_bns,...
                                 'binary', use_binary,...
                                 'mode', bag_mode);
        model.rmse = rms(Ypred - Yvalid);
                                
        if use_bigrams
            % append pre-selector to model
            model.bigram_fs_thresh = BIGRAM_PRE_THRESH;
        end
        
        save(file_name, 'model');
    end
    
    fprintf('Saved model to %s\n', file_name);
    
end

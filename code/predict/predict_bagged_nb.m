function [ Ypred ] = predict_bagged_nb( model, Xtest, metadata )
%PREDICT_BAGGED_NB Predict using bagged multinomial NB classfier
%
% List of parameters for this method:
%
%   'model' Model generated using train_bagged_nb.
%   'test' N x M matrix of testing observations, must be sparse.
%   'metadata' N x 1 cell array of metadata for observations. Unused.
%
% Return values:
%
%   'Ypred' Nx1 matrix of predictions generated using the model.
    
M = size(Xtest,1); % num observations

predictions = zeros(M, model.T);

% iterate over all bags
for t=1:model.T
  
    x = Xtest;
    
    % apply feature selector
    x = model.fs{t}.apply(x);
   
    % predict
    predictions(:,t) = predict_nb(model.bayes{t},x);
    
end

% vote on predictions
if strcmp(model.mode, 'vote')
    Ypred = mode(predictions, 2);
else
    Ypred = mean(predictions, 2);
end
end

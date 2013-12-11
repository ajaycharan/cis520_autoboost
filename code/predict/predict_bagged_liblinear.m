function [ Ypred ] = predict_bagged_liblinear( model, Xtest, metadata )
%PREDICT_BAGGED_LIBLINEAR Predict using bagged liblinear model
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
        
    % apply feature selector
    x = model.fs{t}.apply(Xtest);
   
    % predict
    predictions(:,t) = predict_liblinear(model.linear{t}, x);
    
end

% vote on predictions
if strcmp(model.mode, 'vote')
    Ypred = mode(predictions, 2);
else
    % mode = average
    Ypred = mean(predictions, 2);
end
end


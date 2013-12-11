function [error_out] = cross_validate(observations, labels, k, train_function, predict_function, metadata)
% CROSS_VALIDATE Performs k-fold cross-validation on a model.
%
%   Returns the mean error of k-fold cross-validation on a model built
%   using 'observations' and 'labels', using train_function to build and 
%   predict_function to test the model on each fold.
%
%   Optionally include 'metadata', which will be passed to train_function
%   and predict_function as the 3rd argument.
%
%   USAGE: 
%       
%   [error] = cross_validate(X, Y, 10, @train_nb, @predict_nb)
%
%   The above example uses X = train.counts, Y = train.labels and 10 as the
%   number of folds to use.
%
%   The 'train_function' must have the following signature:
%   
%       function [model] = train(X_train, Y_train)
%
%   The 'predict_function' must have the following signature:
%
%       function [Y_pred] = predict(model, X_test)
%

% seed random number generator with time
rng('shuffle');

% number of observations
N = size(observations, 1);

% generate an array containing the possible fold assignments
fold_asgn = repmat(1:k, 1, ceil(N / k));

% truncate any extra assignments resulting from rounding
fold_asgn = fold_asgn(1:N);

% permute the elements at random
fold_asgn = fold_asgn(randperm(N));

% iterate over all folds
errors = zeros(k,1);

if nargin > 5       % include metadata
    
    parfor i=1:k
    
        % collect training data for k'th fold
        indices_train = fold_asgn ~= i;
        indices_test = ~indices_train;

        X_train = observations(indices_train, :);
        Y_train = labels(indices_train, :);

        X_test = observations(indices_test, :);
        Y_test = labels(indices_test, :);

        meta_train = metadata(indices_train);
        meta_test = metadata(indices_test);

        % train the model WITH metadata
        model = train_function(X_train, Y_train, meta_train);

        % predict the test labels, given training data and test observations
        Y_pred = predict_function(model, X_test, meta_test); 

        % calculate the RMS error for this fold    
        errors(i) = rms(Y_test - Y_pred);

        %fprintf('Finished fold %i\n', i);
    end
    
else                % no metadata
       
    parfor i=1:k
    
        % collect training data for k'th fold
        indices_train = fold_asgn ~= i;
        indices_test = ~indices_train;

        X_train = observations(indices_train, :);
        Y_train = labels(indices_train, :);

        X_test = observations(indices_test, :);
        Y_test = labels(indices_test, :);

        % train the model WITHOUT metadata
        model = train_function(X_train, Y_train);
        
        % predict the test labels, given training data and test observations
        Y_pred = predict_function(model, X_test); 

        % calculate the RMS error for this fold    
        errors(i) = rms(Y_test - Y_pred);

        %fprintf('Finished fold %i\n', i);
    end
end

% average the error
error_out = mean(errors);
end

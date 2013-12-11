function [ Y_pred ] = predict_nb( model, X_test )
% PREDICT_NB Generate predictions using a multinomial NB model.

P_priors = predict_fastnb(model, X_test');
[~, Y_pred] = max(P_priors, [], 2); % assumes labels 1,2,3,4...

end

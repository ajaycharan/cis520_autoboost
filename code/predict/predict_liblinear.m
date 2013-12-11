function Yhat = predict_liblinear(model, Xtest)
% PREDICT_LIBLINEAR Generate predictions using a liblinear model
%
% Predict Yhat for Xtest using liblinear_predict with any liblinear model

N = size(Xtest, 1);
if model.Parameters == 7
    Yhat = liblinear_predict(ones(N, 1), Xtest, model, '-b 1 -q');
else
    Yhat = liblinear_predict(ones(N, 1), Xtest, model, '-q');
end
Yhat(Yhat > 5) = 5;
Yhat(Yhat < 1) = 1;
end
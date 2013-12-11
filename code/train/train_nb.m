function [model] = train_nb(X_train, Y_train)
% TRAIN_NB Train a multinomial naive bayes classifier on observations
% X_train and labels Y_train.

model = train_fastnb(X_train, Y_train, [1 2 3 4 5]);

end

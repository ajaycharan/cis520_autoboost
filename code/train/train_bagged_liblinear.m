function [ model ] = train_bagged_liblinear(Xtrain, Ytrain, metadata, varargin)
%TRAIN_BAGGED_LIBLINEAR Train a liblinear model using bagging.
%
% List of parameters for this method:
%
%   'Xtrain' N x M matrix of training observations, must be sparse.
%   'Ytrain' N x 1 matrix of training labels.
%   'metadata' N x 1 cell array of metadata for observations. Unused.
%
% Return values:
%
%   'model' Model to be used with predict_bagged_liblinear.
%
% Options supported by varargs (defaults below):
%
%   'T'
%       Integer value indicating the number of bags to use.
%   'N'
%       Value in the range (0,1) indicating the fraction of training data
%       to use on each bag.
%   'cost'
%       Cost coeffiecient to use with liblinear. See train_liblinear.
%   'type'
%       Type of liblinear model to train. See train_liblinear.
%   'tresh_bns' 
%       BNS threshold to use when selecting features.
%   'binary' 
%       If true, binary feature selection will be used.
%   'mode'
%       Either 'vote' or 'average'. Describes how the bagged models are
%       combined.

defaults.T = 20;                % number of bags
defaults.N = 0.6;               % fraction of data in each training bag
defaults.cost = 0.23;           % cost parameter for liblinear
defaults.type = 1;              % type parameter for liblinear
defaults.thresh_bns = 1e-3;     % bns threshold
defaults.binary = false;        % use word presence only
defaults.mode = 'vote';         % either 'vote' or 'average'

% parse options
options = propval(varargin, defaults);
if options.T <= 0
    error('T must be a positive integer');
end
if options.N <= 0 || options.N >= 1
    error('N must be in the range (0,1)');
end
valid_modes = {'vote', 'average'};
if ~any(strcmp(options.mode, valid_modes))
    error('Invalid mode selected!');
end

model.T = options.T;
model.N = options.N;
model.linear = cell(model.T, 1);    % where liblinear models are stored
model.mode = options.mode;
model.cost = options.cost;
model.type = options.type;
model.thresh_bns = options.thresh_bns;
model.binary = options.binary;

M = size(Xtrain, 1);  % total size of training data

for t=1:options.T
   
   % generate random subset...
   ind = randperm(M);
   subset_size = floor(M * options.N);
   ind_clip = ind(1:subset_size);
   y = Ytrain(ind_clip);
   
   x = Xtrain(ind_clip,:);
   
   % generate feature selector
   model.fs{t} = FeatureSelector.train(x, y, 'thresh_bns', options.thresh_bns, 'scale', 'bns',...
                                        'binary', options.binary);
   
   % scale by BNS score
   xt = model.fs{t}.apply(x);
   
   % train model
   model.linear{t} = train_liblinear(xt, y, options.type, options.cost);
   
end
end

%% startup.m - Load and configure the Autoboost 9000 environment
%
% Copyright (c) 2013, Chao Qu, Gareth Cross, Pimkhuan Hannanta-Anan. All
% rights reserved.
%   Contact: quchao@seas.upenn.edu, gcross@seas.upenn.edu, pimh@seas.upenn.edu
%
% Loads the Autoboost training environment, and randomly partitions
% training data into a 80% train, 20% validation split.
%
% Description of options available in this file:
%
%   LOAD_METADATA
%       Set to true to load data/metadata.mat
%   COMPILE_LIB_NGRAM
%       Set to true to run libngram/make_mex.m
%   BUILD_BIGRAMS
%       Set to true to generate bigram featurespace
%   BIGRAM_PRE_THRESH
%       BNS threshold applied to bigram featurespace (0.07 recommended)
%
% Note: Loading the environment with BUILD_BIGRAMS consumes almost 4GB of
% memory. Attempting this on a low-end system can cause system lock-up.

% Clear other variables
home;
clearvars -except train quiz metadata quiz_metadata vocab stopword

% we will clock load time
tic;

% Options
% These are the safe defaults
LOAD_METADATA       = true;
COMPILE_LIB_NGRAM   = false;
BUILD_BIGRAMS       = true;
BIGRAM_PRE_THRESH   = 0.07;

present_dir = strsplit(pwd,'/');
if (~strcmp(present_dir(end), 'code'))
    error('!!! Run this script from the ''code'' directory !!!');
end

% Add path
addpath('./libsvm/matlab')
addpath('./liblinear/matlab')
addpath('./utils')
addpath('./model')
addpath('./train')
addpath('./train/unused')
addpath('./support')
addpath('./predict')
addpath('./predict/unused')
addpath('./tune')
addpath('./feature')
addpath('./mex')
addpath('./submission')
addpath('./test')

% Load dataset
if ~exist('quiz', 'var') || ~exist('train', 'var')
    load('../data/review_dataset.mat')
    disp('loading review_dataset')
else
    disp('review_dataset already in workspace.')
end

if LOAD_METADATA
    % Load metadata
    if ~exist('quiz_metadata', 'var') || ~exist('metadata','var')
        disp('loading metadata')
        load('../data/metadata.mat')
        
        % rename this
        metadata = train_metadata;
        clear train_metadata;
    else
        disp('metadata already in workspace.')
    end
end

% partition the training data 80 split
% do this here so BUILD_BIGRAMS can use it too
X = train.counts;
Y = train.labels;
Xtest = quiz.counts;
N = size(X, 1);
ind = randperm(N);
s = ceil(0.8*N);
Xtrain = X(ind(1:s), :);
Ytrain = Y(ind(1:s), :);
Xvalid = X(ind(s+1:end), :);
Yvalid = Y(ind(s+1:end), :);

% partition the metadata along the same boundaries as the data itself
if exist('metadata', 'var')
    fprintf('Partitioning metadata...\n');

    Mtrain = metadata(ind(1:s));
    Mvalid = metadata(ind(s+1:end));
end

fprintf('Data split with %d train data and %d validation data. \n', s, N-s)

% Compile libNGRAM
if COMPILE_LIB_NGRAM
    fprintf('Compiling libngram...\n');
    
    % drop down a directory
    cd 'mex'
    make_mex;
    cd ..
end

% Load digrams
if BUILD_BIGRAMS
    
    if ~exist('Mtrain', 'var') || ~exist('quiz_metadata', 'var') || ~exist('Mvalid', 'var')
        error('!!! Metadata must be loaded if BUILD_DIGRAMS=1 !!!');
    end
    
    if exist('create_ngram_tree','file')~=3
        error('!!! MEX files for libNGRAM have not been compiled !!!');
    end
    
    % unfortunately these ought to be rebuilt every time we change the partition
    % this takes a few seconds
    
    fprintf('Collecting unigrams...\n');
    unigrams_train = get_all_review_texts(Mtrain);
    unigrams_valid = get_all_review_texts(Mvalid);
    unigrams_all = get_all_review_texts(metadata);
    
    fprintf('Generating bi-gram trees...\n');
    [tree_train, size_train] = create_ngram_tree(unigrams_train);
    [tree_valid, size_valid] = create_ngram_tree(unigrams_valid);
    [tree_all, size_all] = create_ngram_tree(unigrams_all);
    
    fprintf('Pulling vocab and feature spaces...\n');
    
    % this is quick, just do it every time
    vocab_train = get_tree_vocab(tree_train);
    vocab_valid = get_tree_vocab(tree_valid);
    vocab_bigrams = get_tree_vocab(tree_all);
    
    % training and validation
    Xd_train = get_tree_featurespace(tree_train, numel(Mtrain), size_train);
    Xd_valid = get_tree_featurespace(tree_valid, numel(Mvalid), size_valid);
    [indices_train_in_valid, indices_valid_in_train] = tree_cull_vocab(tree_train, tree_valid);
    
    % threshold away junk
    % note that bigrams are hardcoded to binary = true !
    FS_bigram_train = FeatureSelector.train(Xd_train, Ytrain,... 
                                            'thresh_bns', BIGRAM_PRE_THRESH,...
                                            'scale', 'none');
    Xd_train = FS_bigram_train.apply(Xd_train);
    vocab_train = vocab_train(FS_bigram_train.feat_ind);
    
    % full set
    Xd = get_tree_featurespace(tree_all, numel(metadata), size_all);
    
    % do the same for full
    FS_bigram_all = FeatureSelector.train(Xd, Y,... 
                                            'thresh_bns', BIGRAM_PRE_THRESH,...
                                            'scale', 'none');
                            
    Xd = FS_bigram_all.apply(Xd);
    vocab_bigrams = vocab_bigrams(FS_bigram_all.feat_ind);
    
    % save this vocab
    % necessary for model!!
    fprintf('Writing vocab_bigrams.mat to disk!\n');
    save('./model/vocab_bigrams.mat', 'vocab_bigrams');
end

% calculate memory usage
S = whos;
memory_usage = 0;
for i=1:numel(S)
    memory_usage = memory_usage + S(i).bytes;
end

fprintf('\nAutoboost 9000 loaded in %.2f seconds. Bytes used: %i (%.2f GB)\n\n', toc, memory_usage, memory_usage / (1024^3));
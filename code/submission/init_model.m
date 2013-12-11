function [model] = init_model(~)
% INIT_MODEL Load all model files

% Add paths necessary for execution
addpath('./liblinear/matlab')
addpath('./utils')
addpath('./model')
addpath('./support')
addpath('./predict')
addpath('./feature')
addpath('./libngram')

% Location of bigram vocabulary
DICTIONARY_FILE = 'vocab_bigrams';

% Location of ensemble coefficients
ENSEMBLE_FILE = 'ensemble.mat';

% Models generated using lib-linear
BAGGED_LIBLINEAR_MODEL_FILES = {'l2rl2ld_svm_bag', 'l2rd_lr_bag', 'l2rl1ld_svm_bag', ...
     'l1r_lr_bag', 'l2rl2ld_svm_bi_bag', 'l1r_lr_bi_bag', 'l2rl1ld_svm_bi_bag'};
 
% Models generated using Naive Bayes
BAGGED_NB_MODEL_FILES = {'nb_mn_bag', 'nb_mn_bi_bag'};

model = struct();
model.bagged_liblinear_models = cell(numel(BAGGED_LIBLINEAR_MODEL_FILES),1);
model.bagged_nb_models = cell(numel(BAGGED_NB_MODEL_FILES),1);

tic;

% Load vocab
vocab = load(DICTIONARY_FILE);
model.vocab_bigrams = vocab.vocab_bigrams;

% Load ensemble coefficients
coef = load(ENSEMBLE_FILE);
model.w = coef.w;

model_count = 0;

% Load liblinear models
for n=1:numel(BAGGED_LIBLINEAR_MODEL_FILES)
    
    bagged = load(BAGGED_LIBLINEAR_MODEL_FILES{n});
    model.bagged_liblinear_models{n} = bagged.model;
    model.bagged_liblinear_models{n}.needs_bigrams = isfield(bagged.model, 'bigram_fs_thresh');
    
    model_count = model_count+1;
end

% Load NB models
for n=1:numel(BAGGED_NB_MODEL_FILES)
    
    bagged = load(BAGGED_NB_MODEL_FILES{n});
    model.bagged_nb_models{n} = bagged.model;
    model.bagged_nb_models{n}.needs_bigrams = isfield(bagged.model, 'bigram_fs_thresh');
    
    model_count = model_count+1;
end

model.count = model_count;

fprintf('== Autoboost 9000 loaded %i models in %.3f seconds ==\n', model_count, toc);
end

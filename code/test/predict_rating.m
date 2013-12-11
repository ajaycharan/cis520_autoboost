function Yvalid = predict_rating(~, Xvalid, ~, meta_quiz, ~)
%PREDICT_RATING Returns the predicted ratings, given wordcounts and 
% additional features.
%
% Usage:
%
%   RATES = PREDICT_RATING(Xtrain, Xvalid, meta_train, ...
%                         meta_quiz, Ytrain);
%
% This is the function that we will use for checkpoint evaluations and the
% final test. It takes a set of wordcount and additional features and produces a
% ranking matrix as explained in the project overview.
%
% This method operates in much the same way as tune_ensemble.m. However, it
% does not support unbagged models, since they do not appear in the final
% submission.

% Models generated using bagged liblinear
BAGGED_LIBLINEAR_MODEL_FILES = {'l2rl2ld_svm_bag', 'l2rd_lr_bag', 'l2rl1ld_svm_bag', ...
     'l1r_lr_bag', 'l2rl2ld_svm_bi_bag', 'l1r_lr_bi_bag', 'l2rl1ld_svm_bi_bag'};
 
% Models generated using bagged bayes
BAGGED_NB_MODEL_FILES = {'nb_mn_bag', 'nb_mn_bi_bag'};

% Load ensemble coeffients
w = load('./model/ensemble.mat');
w = w.w;

Yvalid_hat = [];    % quiz predictions for all models

vocab_bigrams = load('./model/vocab_bigrams.mat');
vocab_bigrams = vocab_bigrams.vocab_bigrams;    % get from struct

unigrams_quiz = get_all_review_texts(meta_quiz);
[tree_quiz, Vsize_quiz] = create_ngram_tree(unigrams_quiz);

[idx_into_quiz, idx_into_vocab] = tree_cull_vocab_v(tree_quiz, vocab_bigrams);

% get feature space of bigrams
Xb_quiz = get_tree_featurespace(tree_quiz, numel(meta_quiz), Vsize_quiz);
XV = sparse(size(Xvalid,1), numel(vocab_bigrams));
XV(:,idx_into_vocab) = Xb_quiz(:, idx_into_quiz);

%% Process bagged liblinear models
for n=1:numel(BAGGED_LIBLINEAR_MODEL_FILES)
   
    bagged = load(BAGGED_LIBLINEAR_MODEL_FILES{n});
    bagged = bagged.model; % pull from loaded struct
    tic;
    
    if isfield(bagged, 'bigram_fs_thresh')
        XVal = XV;
    else
        XVal = Xvalid;
    end
                                
    Yvalid_n = predict_bagged_liblinear(bagged, XVal, meta_quiz);
    
    Yvalid_hat = [Yvalid_hat Yvalid_n];
        
    fprintf('-- Ran model %s (%f seconds) --\n', BAGGED_LIBLINEAR_MODEL_FILES{n}, toc);
        
end

for n=1:numel(BAGGED_NB_MODEL_FILES)
   
    bagged = load(BAGGED_NB_MODEL_FILES{n});
    bagged = bagged.model; % pull from loaded struct
    tic;
                   
    if isfield(bagged, 'bigram_fs_thresh')
        XVal = XV;
    else
        XVal = Xvalid;
    end
    
    Yvalid_n = predict_bagged_nb(bagged, XVal, meta_quiz);
    
    Yvalid_hat = [Yvalid_hat Yvalid_n];

    fprintf('-- Ran model %s (%f seconds) --\n', BAGGED_NB_MODEL_FILES{n}, toc);
    
end

%% Ensemble
Yvalid = Yvalid_hat * w;

fprintf('\n===============================');
fprintf('\n ** FINISHED PREDICTIONS **\n');
fprintf('===============================\n');

end

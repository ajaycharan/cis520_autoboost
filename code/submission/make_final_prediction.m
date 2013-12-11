function prediction = make_final_prediction(model, test_words, test_meta)
% MAKE_FINAL_PREDICTION
%   test_words : a 1xp vector representing "1" test sample.
%   test_meta : a struct containing the metadata of the test sample.
%   model : what you initialized from init_model.m
%
%   Output
%   prediction : a scalar which is your prediction of the test sample
%
% **Note: the function will only take 1 sample each time.

% Extract bigrams
[item_tree, num_bigrams] = create_ngram_tree({test_meta.text});
if ~isstruct(item_tree)
    % no bigrams available for this observation
    no_bigrams = true;
else
    no_bigrams = false;
   
    % get the feature space
    Xb_org = get_tree_featurespace(item_tree, 1, num_bigrams);
       
    % map to model feature space
    [idx_into_quiz, idx_into_vocab] = tree_cull_vocab_v(item_tree, model.vocab_bigrams);
    
    Xb = zeros(1, numel(idx_into_vocab));
    Xb(1,idx_into_vocab) = Xb_org(:,idx_into_quiz);
    Xb = sparse(Xb);
end

Yhat = [];

for n=1:numel(model.bagged_liblinear_models)
    mdl = model.bagged_liblinear_models{n};
    
    if mdl.needs_bigrams
        if no_bigrams
            continue;
        end
        X = Xb;
    else
        X = test_words;
    end
    
    Yhat(end+1) = predict_bagged_liblinear(mdl, X, {});
end

for n=1:numel(model.bagged_nb_models)
    mdl = model.bagged_nb_models{n};
    
    if mdl.needs_bigrams
        if no_bigrams
            continue;
        end
        X = Xb;
    else
        X = test_words;
    end
    
    Yhat(end+1) = predict_bagged_nb(mdl, X, {});
end

if numel(Yhat) == numel(model.w)
    prediction = Yhat * model.w; % we have the right number of models to do regression
else
    prediction = mean(Yhat);     % missing a column - just average
end
end

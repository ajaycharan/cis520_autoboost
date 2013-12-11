%% RUN_SUBMISSION - Load environment, generate quiz answers
startup;

Xt_counts = train.counts;
Yt = train.labels;
Xq_counts = quiz.counts;

% metadata is unused
if ~exist('metadata', 'var') || ~exist('quiz_metadata', 'var')
    error('!! Metadata must be loaded in order to execute run_submission.m !!');
end

Xt_additional_features = metadata;
Xq_additional_features = quiz_metadata; % CHANGE ME IF VALIDATING!

%% Run algorithm
rates = predict_rating(Xt_counts, Xq_counts, Xt_additional_features, Xq_additional_features, Yt);

%% Save results to a text file for submission
fprintf('\n!! Generating submit.txt !!\n');
dlmwrite('submit.txt', rates, 'precision', '%d');

%% SIMULATE_SUBMISSION - Simulate a submission to ensure all components function
% This script is intended to be run from the deployed folder on BigLab

fprintf('Loading data for test...\n');
data = load('review_dataset.mat');
metadata = load('metadata.mat');

X = data.train.counts;
Y = data.train.labels;
XT = X';

meta = metadata.train_metadata;

fprintf('Running init_model()\n');
model = init_model(meta);

fprintf('Generating 5000 predictions...\n');

T = CTimeleft(5000);
Ypred = zeros(5000,1);
for i=1:5000
    T.timeleft();
    
    Ypred(i) = make_final_prediction(model,XT(:,i)',meta(i));
    
end

fprintf('RMSE on training data: %f\n', rms(Ypred - Y(1:5000)));

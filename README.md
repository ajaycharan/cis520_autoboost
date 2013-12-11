cis520_final
============

Description:

Sentiment analysis for Yelp. This project scored a final RMSE of 0.7615, taking first place
out of 47 teams in the CIS520 final competition. Our solution, 'Autoboost', uses a collection
of bagged models, along with both unigram and bigram features to deliver best performance. 

A brief report on the model and our findings is available in report.pdf. Note that this is
an in-class report - not written for external publication. 

Tournament rankings (top 5):

    1) Autoboost					0.761581
    2) Super Vital Machine			0.782488
    3) Aeroducks					0.793093
    4) PENNalizers					0.812594
    5) X							0.812632

License: Apache 2.0, see LICENSE for details.

Instructions:

The following steps will train and validate the model, generating RMSE scores for each individual
model as well as the final ensemble:

	1) cd into the 'mex' folder and run make_mex.m
	2) Ensure 'BUILD_BIGRAMS' is true in startup.m
	3) Run tune_ensemble.m and wait for a few minutes.

Checkpoints:

[X]   Nov. 20 Submit group.txt (1%)

[X]   Nov. 22 Beat 1st baseline quiz RMSE 0f 1.30 (9%)

[X]   Dec. 03 Beat 2nd baseline quiz RMSE of 1.00 (20%)

[X]   Dec. 06 Submit final classifier for competition as well as all other implementations (50%)

[X]   Dec. 10 Submit final report (20%)

Directory structure:

├─── data
	├─── review_dataset.mat Training data (courtesy of Yelp) used for the contest  
    └─── metadata.mat 		Observation metadata associated with review_dataset  
├─── code  
    ├─── deploy.sh 		   	Script to copy only components of model for final submission  
    ├─── feature 			Feature selection and analysis  
    ├─── group.txt			Name of the group for leaderboard submission  
    ├─── liblinear 			Binaries/code for MATLAB liblinear implementation  
    ├─── mex 				C++ source for MEX implementation of ngrams + misc other MEX code  
    ├─── libsvm 			Binaries for MATLAB libsvm implementation  
    ├─── model 				MAT files of compiled models  
	├─── predict			Prediction methods for sub-models  
	├─── startup.m			Loads the Autoboost 9000 environment with necessary data  
	├─── submission 		Code for running the final submission  
    ├─── support			Various supporting methods  
    ├─── test				Code for running the quiz submission (test set)  
    ├─── train				Training methods for sub-models  
    ├─── tune				Tuning methods for sub-models + ensembles  
    └─── utils				Misc. scripts for testing/evaluating data  

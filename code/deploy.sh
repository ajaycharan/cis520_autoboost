#!/bin/sh

# 
# deploy.sh
# Copies components of project necessary for submission into a new folder 'deploy'
#

echo 'Deploying submission components...'

mkdir deploy

# Group file
cp group.txt ./deploy
cp README.md ./deploy
cp startup.m ./deploy

# MATLAB code + MAT files
cp -rv ./support ./deploy

cp -rv ./submission/* ./deploy/

cp -rv ./feature ./deploy
cp -rv ./predict ./deploy

cp -rv ./model ./deploy
rm -rf ./deploy/model/old

# libngram code + compile script, but not binaries
cp -rv ./libngram ./deploy
rm -rf ./deploy/libngram/*.mex*

# liblinear
cp -rv ./liblinear ./deploy

# These are not necessary to run the model, included for checking by TAs
cp -rv ./train ./deploy
cp -rv ./tune ./deploy
cp -rv ./utils ./deploy

# Only enable these for testing!!
#cp -rv ../data/review_dataset.mat ./deploy
#cp -rv ../data/metadata.mat ./deploy

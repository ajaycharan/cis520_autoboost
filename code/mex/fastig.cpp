/*
 * fastig.cpp
 *
 *  Copyright 2013 Gareth Cross
 * 
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 * 
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 * 
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *	Created on: 12/1/2013
 *		Author: Gareth Cross
 */

extern "C" {
    #include "mex.h"
    #include "stdint.h"
    #include "string.h"
}

#include <vector>
#include <cmath>

/**
 *  @brief IG = fastig(X,Y,colidx,labels) calculates the information gain of X and Y
 *  @param X Matrix where rows are observations, columns are features (M x N)
 *  @param Y Rows are labels for observations (M x 1)
 *  @param colidx Indices of columns in X to calulate on (1 x J), where J is at most N
 *  @param labels Row vector (1 x K) of labels in Y to consider. ie [1 2 3 4 5] for 5-star ranking
 *  @return Returns a 1 x J row vector of information gains
 */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{
    //  check inputs
    if(nrhs != 4)
    {
        mexErrMsgIdAndTxt("fastig:invalidNumInputs",
                          "Four inputs required.");
    }
    else if (nlhs != 1)
    {
        mexErrMsgIdAndTxt("fastig:invalidNumOutputs",
                          "One output required.");
    }
    else if(!mxIsSparse(prhs[0]))
    {
        mexErrMsgIdAndTxt("fastig:inputNotSparse",
                          "First input must be sparse.");
    }
    else if (!mxIsDouble(prhs[0])|| !mxIsDouble(prhs[1]) || !mxIsDouble(prhs[2]) || !mxIsDouble(prhs[3]))
    {
        mexErrMsgIdAndTxt("fastig:inputNotNumeric",
                          "All inputs must be of type double");
    }
    
    const mxArray * X = prhs[0];
    const mxArray * Y = prhs[1];
    const mxArray * indices = prhs[2];
    const mxArray * labels = prhs[3];
    
    const mwSize M = mxGetM(X);
    const mwSize N = mxGetN(X);
    const mwSize J = mxGetN(indices);
    const mwSize K = mxGetN(labels);
    
    //  check dimensions
    if (mxGetM(Y) != M) {
        mexErrMsgIdAndTxt("fastig:invalidInputDimensions",
                          "X and Y must have same number of rows.");
    }
    if (mxGetN(Y) != 1 || !M) {
        mexErrMsgIdAndTxt("fastig:invalidInputDimensions",
                          "Y should be a M x 1 vector (M > 0).");
    }
    if (mxGetM(indices) != 1 || !J) {
        mexErrMsgIdAndTxt("fastig:invalidInputDimensions",
                          "Indices should be 1 x N vector (N > 0).");
    }
    if (mxGetM(labels) != 1 || !K) {
        mexErrMsgIdAndTxt("fastig:invalidInputDimensions",
                          "Labels should be 1 x K vector (K > 0).");
    }
    
    //  feature space data
    const mwIndex * xIR = mxGetIr(X);
    const mwIndex * xJC = mxGetJc(X);
    
    //  index data
    const double * indexData = static_cast<double*>( mxGetData(indices) );
    
    //  label data
    const double * yData = static_cast<double*>( mxGetData(Y) );
    const double * labelData = static_cast<double*>( mxGetData(labels) );
    
    //  space for result
    mxArray * IG = mxCreateNumericArray(0,0,mxDOUBLE_CLASS, mxREAL);
    double * igData = static_cast<double*> ( mxMalloc(J * sizeof(double)) );
    
    //  create matrix of occurrences of labels
    std::vector<bool> Z(M*K, false);
    std::vector<int> Z_sum(K, 0);
    
#define Z_ACC(i,k)  Z[(i)*K + (k)]
    
    for (mwIndex i=0; i < M; i++)
    {
        for (mwIndex k=0; k < K; k++)
        {
            if (yData[i] == labelData[k])
            {
                Z_ACC(i, k) = true;
                Z_sum[k]++;
            }
        }
    }
    
    //  calculate entropy of Y
    double H = 0.0;
    for (mwIndex k=0; k < K; k++)
    {
        double p = Z_sum[k] / (M*1.0);
        if (Z_sum[k] > 0) {
            H += -p*log2(p);
        }
    }
    
    //  iterate over selected features in input
    double integral;
    for (mwIndex j=0; j < J; j++)
    {
        mwIndex col_idx;
        if (std::modf(indexData[j], &integral) != 0.0) {
            mxDestroyArray(IG);
            mexErrMsgIdAndTxt("fastig:invalidInputValues",
                              "Column indices must have only integer parts.");
        }
        col_idx = static_cast<mwIndex>(integral);
        
        if (col_idx == 0 || col_idx > N) {
            mxDestroyArray(IG);
            mexErrMsgIdAndTxt("fastig:invalidInputValues",
                              "Column indices must be in the range of [1,N]");
        }
        col_idx--;  //  convert to C format
        
        mwIndex rowStart = xJC[col_idx];
        mwIndex rowEnd = xJC[col_idx+1];
        if (rowStart == rowEnd) {
            //  empty column of X
            igData[j] = 0.0;
            continue;
        }
        rowEnd--;   //  last index in this column
        
        double cond_H = 0.0;  //  conditional entropy
        
        //  iterate over labels
        for (mwIndex k=0; k < K; k++)
        {
            int y_and_x=0, y_and_notx=0;
            unsigned long nnz=0, nz=0;
            
            //  iterate over rows of X, only those with non-zero values
            for (mwIndex i=rowStart; i <= rowEnd; i++)
            {
                mwIndex rowidx = xIR[i];    //  index of row
                
                if (Z_ACC(rowidx, k) == true) {
                    y_and_x++;
                }
            }
            
            nnz = rowEnd - rowStart + 1;            //  number of non zero observations in X
            nz = M - nnz;                           //  number of zero observations in X
            y_and_notx = Z_sum[k] - y_and_x;        //  number of times y appears but X does not
            
            const double p_y_given_x = y_and_x / (nnz*1.0);
            const double p_y_given_notx = y_and_notx / (nz*1.0);
            const double px = nnz / (M*1.0);        //  probability of seeing this feature
        
            if (y_and_x > 0) {
                cond_H += -p_y_given_x * log2(p_y_given_x) * px;
            }
            if (y_and_notx > 0) {
                cond_H += -p_y_given_notx * log2(p_y_given_notx) * (1.0 - px);
            }
        }
        
        igData[j] = H - cond_H; //  information gain
    }
    
    //  done
    mxSetData(IG, igData);
    mxSetM(IG, 1);
    mxSetN(IG, J);
    
    //  done
    plhs[0] = IG;
}

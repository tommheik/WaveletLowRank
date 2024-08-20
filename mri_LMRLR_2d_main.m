%%%%%% mri_LMRLR_2d_main.m %%%%%%
%
% Example code for reconstructing the MRI data using local
% multiresolution low-rank approximation
% 
%%%%%%
%
% References:
% [Low-rank + sparse decomposition of dynamic MRI (with the associated 
% forward operator and Fourier domain sparsity instead of wavelets and CT)]
% Otazo, R., Candes, E., & Sodickson, D. K. (2015)
% "Low‐rank plus sparse matrix decomposition for accelerated dynamic MRI
% with separation of background and dynamic components."
% Magnetic resonance in medicine, 73(3), 1125-1136.
% doi: 10.1002/mrm.25240.
% Original code: https://cai2r.net/resources/ls-reconstruction-matlab-code/
% 
% [RPCA applied to 4D( 3D + time) tomography]
% Gao, H., Cai, J. F., Shen, Z., & Zhao, H. (2011).
% "Robust principal component analysis-based four-dimensional computed
% tomography."
% Physics in Medicine & Biology, 56(11), 3181.
% 
%%%%%%
% 
% Requirements:
% Wavelet Toolbox
% https://mathworks.com/products/wavelet.html
% 
% l+s_mri_v1 data and codes
% Ricardo Otazo
% https://cai2r.net/resources/ls-reconstruction-matlab-code/
%
%%%%%%
%
% Created 20.10.2023 - Last edited 20.10.2023
% Tommi Heikkilä
% University of Helsinki

% Clear workspace
clear all
close all

%% Load data

% load undersampled data 
addpath(genpath(Gpath('/l+s_mri_v1')));
load('cardiac_perf_R8.mat');
[nx,ny,nt,nc]=size(kdata);

addpath('./util')
%% Choose parameters and operators

% L+S reconstruction ******************************************************
A = Emat_xyt(kdata(:,:,:,1)~=0,b1); % Forward operator
param.maxIter = 200;
param.tol = 5e-4;
param.mu = 1e-2; % Regularization parameter
param.plotFreq = 10; % Visualize iterations 
param.wName = 'db2';
param.wLevel = 5;
param.wMode = 'sym';
xSz = [nx,ny,nt];

%% Run the algorithm
% Iterate
[recn, iter, info] = LMRLRalgorithm_mri(A, kdata, xSz, param);

%% Look at outcome

figure(200);
montage(abs(recn),'DisplayRange', [])
title('Final reconstruction')
drawnow
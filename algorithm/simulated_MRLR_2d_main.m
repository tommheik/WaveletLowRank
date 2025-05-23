%%%%%% stempo_MRLR_2d_main.m %%%%%%
%
% Example code for reconstructing the simulated data using
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
% ASTRA Toolbox
% https://www.astra-toolbox.com/
% Recommended v1.9 or higher
%
% Spot - A Linear-Operator Toolbox
% https://www.cs.ubc.ca/labs/scl/spot/
% Recommended v1.2
%
% Wavelet Toolbox
% https://mathworks.com/products/wavelet.html
%
%%%%%%
%
% Created 24.9.2023 - Last edited 28.2.2025
% Tommi Heikkilä
% LUT University

% Clear workspace
clear all
close all

%% Load data

load("../data/simData_dynamic_256x256x180_parallel.mat");

addpath('../util')
addpath('./util')
%% Choose parameters

N = size(obj,1); % Spatial resolution of 2d slices
[numberImages, Ndet] = size(sinogram);

% We wish to split the sinogram into as many
% time steps as possible (since that greatly limits the SVD and number of
% singular values). We can take 'Nangles' projections per time step and 
% advance only by angShift projections for the next time step such that two 
% consecutive projections share 'Nangles - angShift' projections.
% 
% For example: (Nangles = 24, angShift = 4)
% p   :  1   2   3   4   5  ... 24   25   26   27   28   29 ... 360
% t=1 :  X   X   X   X   X  ...  X
% t=2 :                  X  ...  X    X    X    X    X
% etc.
Nangles = 30;
angShift = 10;
T = (numberImages - Nangles + angShift) / angShift;

% Projection angles are stored in columns (degrees!)
angleArray = ((0:1:Nangles-1)' + angShift*(0:1:T-1));

% Reorganize the data in a similar manner to match the projection angles
mInd = (1:Nangles)' + angShift*(0:1:T-1);
m = permute(reshape(sinogram(mInd(:),:),[Nangles,T,Ndet]),[1,3,2]);
% Permuting the array guarantees the time steps stay in order once m is
% dropped into a single column vector

% Visualize data
if false
    figure(1)
    for t = 1:T
        imagesc(m(:,:,t)')
        title(sprintf('Data m_{%d}', t))
        colormap gray
        axis off
        drawnow
        pause(0.02)
    end
end

%% Forward operator

% Build a block diagonal forward operator
opCell = cell(1,T);
Anorm = zeros(1,T);
for t = 1:T
    % Create and store the operator in a cell array
    fprintf("Op. %i/%i: ", t, T);
    anglesRad = deg2rad(angleArray(:,t));
    opCell{t} = create_ct_operator_2d_parallel_astra_cpu(N, N, Ndet, anglesRad);
    Anorm(t) = normest(opCell{t});
end
% cell{:} gives the content of a cell array as comma separated list
A = blkdiag(opCell{:});
Anorm = max(Anorm);

% Normalize data and operator
A = A/Anorm;
m = m(:)/Anorm;

%% Run the algorithm
% Set parameters

Psz = [16, 16; 8, 8; 8, 8; 8, 8];

param.Psz = Psz;
param.maxIter = 500;
param.tol = 5e-4;
param.mu = 1e-1; %1e-3; % Regularization parameter
param.plotFreq = 10; % Visualize iterations 
param.wName = 'db2';
param.wLevel = 3;
param.wMode = 'per';
xSz = [N,N,T];

% Iterate
[recn, iter, info] = LMRLRalgorithm(A, m, xSz, param);

%% Look at outcome

figure(200);
montage(recn,'DisplayRange', [])
title('Final reconstruction')
drawnow
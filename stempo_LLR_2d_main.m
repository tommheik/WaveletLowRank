%%%%%% stempo_LLR_2d_main.m %%%%%%
%
% Example code for reconstructing the STEMPO phantom data using local
% low-rank approximation
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
% HelTomo Toolbox
% https://github.com/Diagonalizable/HelTomo
% v2.0
%
%%%%%%
%
% Created 16.6.2023 - Last edited 19.6.2023
% Tommi Heikkilä
% University of Helsinki

% Clear workspace
clear all
close all

%% Load data

binning = 8;
dataset = 'cont360'; % 'seq8x45'; % cont360
switch dataset
    case 'seq8x45'
        load(Gpath(sprintf('DynamicCTphantom/v3/ZENODO/stempo_seq8x45_2d_b%d.mat',binning)))
        dAngle = 8; % seq8x45 data is measured every 8 degrees
    case 'cont360'
        load(Gpath(sprintf('DynamicCTphantom/v3/ZENODO/stempo_cont360_2d_b%d.mat',binning)))
        dAngle = 1; % cont360 data is measured every 1 degrees
end

addpath('./util')
%% Choose parameters

N = 2240 / binning; % Spatial resolution of 2d slices

% We wish to split the 8 full rotations (45 projections each) into as many
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
T = (CtData.parameters.numberImages - Nangles + angShift) / angShift;

% Projection angles are stored in columns
angleArray = dAngle*((0:1:Nangles-1)' + angShift*(0:1:T-1));

% Reorganize the data in a similar manner to match the projection angles
mInd = (1:Nangles)' + angShift*(0:1:T-1);
m = permute(reshape(CtData.sinogram(mInd(:),:),[Nangles,T,N]),[1,3,2]);
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
for t = 1:T
    % Change the projection angles stored in CtData
    CtData.parameters.angles = angleArray(:,t);
    % Create and store the operator in a cell array
    fprintf("Op. %i/%i: ", t, T);
    opCell{t} = create_ct_operator_2d_fan_astra(CtData, N, N);
end
% cell{:} gives the content of a cell array as comma separated list
A = blkdiag(opCell{:});

% Normalize data and operator
Anorm = normest(A);
A = A/Anorm;
m = m(:)/Anorm;

mMax = max(m(:));
delta = 0.03; % Noise level
m = m + delta*mMax*randn(size(m)); % Gaussian noise
fprintf('Added Gaussian noise: delta = %0.2f\n',delta)

%% Run the algorithm
% Set parameters
param.maxIter = 500;
param.tol = 5e-4;
param.mu = 1e-2; % Regularization parameter
param.pSz = 7*[1 1]; % p x p patches
param.plotFreq = 20; % Visualize iterations 
xSz = [N,N,T];

% Iterate
[recn, iter, info] = LLRalgorithm(A, m, xSz, param);

%% Look at outcome

figure(200);
montage(recn,'DisplayRange', [])
title('Final reconstruction')
drawnow

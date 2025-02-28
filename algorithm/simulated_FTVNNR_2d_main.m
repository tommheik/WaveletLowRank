%%%%%% stempo_LMRLR_2d_main.m %%%%%%
%
% Example code for reconstructing the simulated data using Fast TV +
% Nuclear Norm Regularization (FTVNNR)
% 
%%%%%%
%
% References:
% [1] Jiawen Yao, Zheng Xu, Xiaolei Huang, Junzhou Huang  
% " An Efficient Algorithm for Dynamic MRI Using Low-Rank and Total Variation Regularizations",
% In Medical Image Analysis, 44, 14-27, 2018.
% [2] Jiawen Yao, Zheng Xu, Xiaolei Huang, Junzhou Huang 
% "Accelerated dynamic MRI reconstruction with total variation and nuclear norm regularization", 
% In MICCAI 2015.
% 
% The code also include partial useful functions from SSMRI MATLAB Toolbox
% http://web.engr.illinois.edu/~cchen156/SSMRI.html
%
% By Jiawen Yao@UT Arlington
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
% Created 25.2.2025 - Last edited 25.2.2025
% Tommi Heikkilä
% LUT University

% Clear workspace
clear all
close all

%% Load data

load("data/simData_all_256x256x180_parallel.mat");

addpath('./util')
if isfolder('./FTVNNR_Dynamic_MRI_MEDIA-master')
    addpath(genpath('./FTVNNR_Dynamic_MRI_MEDIA-master'))
else
    error("Can not find the FTVNNR master folder, did you download it from\n 'https://github.com/uta-smile/FTVNNR_Dynamic_MRI_MEDIA/tree/master'?")
end
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

Nangles = 19;
angShift = 11;
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
Anorm = max(Anorm);
for t = 1:T; opCell{t} = opCell{t}/Anorm; end % Normalize operators
% Use the same Casorati matrix formulation as in the original FTVNNR
A = A_operator(@(X) A_fwd(opCell, X, [Nangles, Ndet]), @(Y) A_adj(opCell, Y, [N, N]));

% Normalize data
m = m / Anorm;

%% Run the algorithm
% Set parameters

tic
param.lambda_1 = 0.001; % TV reg. param
param.lambda_2 = 5; % Nuclear norm reg. param
param.max_iter = 500;
param.tol = 5e-4;
param.verbose = 1;  % It prints out evaluation each step, may set to 0 for faster speed, requires ground truth
param.L = 1; % Largest eigenvalue of A^T A
param.t1 = 4; % "Iteration step size"
param.debug_output = 0;

% Pick one angle to represent each time step
avgAngle = angleArray(round(Nangles/2), :);

objInd = round(size(obj,3) * avgAngle / numberImages);
ref = obj(:,:,objInd);

% Iterate
xSz = [N, N, T];
X = TVLR_opt(A, m, param, ref);
recn = reshape(X, xSz);
time_FTVNNR = toc;

%% Look at outcome

figure(200);
montage(recn,'DisplayRange', [])
title('Final reconstruction')
drawnow

%%

function Y = A_fwd(opCell, X, sinoSz)
% Forward operator call, X is N x N x T, output is m x n x T
vec = @(x) x(:);
T = size(X,3);
Y = zeros([sinoSz, T], class(X));
for t = 1:T
    Y(:,:,t) = reshape(opCell{t}*vec(X(:,:,t)), sinoSz);
end
end

function X = A_adj(opCell, Y, imSz)
% Adjoint call, Y is m x n x T, output is N x N x T
vec = @(x) x(:);
T = size(Y,3);
X = zeros([imSz, T], class(Y));
for t = 1:T
    X(:,:,t) = reshape(opCell{t}'*vec(Y(:,:,t)), imSz);
end
end
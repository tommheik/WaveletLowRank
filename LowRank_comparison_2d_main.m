%%%%%% stempo_LMRLR_2d_main.m %%%%%%
%
% Example code for reconstructing dynamic CT data using different low-rank
% methods, such as
%   Local Low Rank
%   Low rank + sparse
%   Local Multiresolution Low rank
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
% Wavelet Toolbox
% https://mathworks.com/products/wavelet.html
%
%%%%%%
%
% Created 15.6.2023 - Last edited 28.11.2024
% Tommi Heikkilä
% LUT University

% Clear workspace
clearvars
close all

visualize = false;

%% Load data

dataType = 'STEMPO'; % 'STEMPO' or 'simulated';

switch dataType
    case 'STEMPO'
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
        N = 2240 / binning; % Spatial resolution of 2d slices
        Ndet = N;
        numberImages = CtData.parameters.numberImages;
        sinogram = 100*CtData.sinogram; % Magic number

    case 'simulated'
        load("data/simData_all_256x256x180_parallel.mat");

        N = size(obj,1); % Spatial resolution of 2d slices
        [numberImages, Ndet] = size(sinogram);
        dAngle = 1; % Difference between consecutive projection angles
end

addpath('./util')

%% Choose parameters

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
angleArray = dAngle*((0:1:Nangles-1)' + angShift*(0:1:T-1));

% Reorganize the data in a similar manner to match the projection angles
mInd = (1:Nangles)' + angShift*(0:1:T-1);
m = permute(reshape(sinogram(mInd(:),:),[Nangles,T,Ndet]),[1,3,2]);
% Permuting the array guarantees the time steps stay in order once m is
% dropped into a single column vector

% Visualize data
if visualize
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

x0 = zeros([N,N,T]);

%% Forward operator

% Build a block diagonal forward operator
opCell = cell(1,T);
Anorm = zeros(1,T);
for t = 1:T
    % Create and store the operator in a cell array
    fprintf("Op. %i/%i: ", t, T);
    switch dataType
        case 'STEMPO'
            % Change the projection angles stored in CtData
            CtData.parameters.angles = angleArray(:,t);
            opCell{t} = create_ct_operator_2d_fan_astra(CtData, N, N);
        case 'simulated'
            anglesRad = deg2rad(angleArray(:,t));
            opCell{t} = create_ct_operator_2d_parallel_astra_cpu(N, N, Ndet, anglesRad);
    end
    Anorm(t) = normest(opCell{t});
end
% cell{:} gives the content of a cell array as comma separated list
A = blkdiag(opCell{:});
Anorm = max(Anorm);

% Normalize data and operator
A = A/Anorm;
m = m(:)/Anorm;

% Add extra noise? 
mMax = max(m(:));
delta = 0.0; % Noise level
m = m + delta*mMax*randn(size(m)); % Gaussian noise
fprintf('Added Gaussian noise: delta = %0.2f\n',delta)

tol = 5e-4; % Stopping tolerance

%% Run the LMRLR algorithm
% Set parameters
switch dataType
    case 'simulated'
        Psz = [64, 64; 64, 64; 64, 64]; % Patch sizes (per scale)
        mu = 1e0; % Regularization parameter
    case 'STEMPO'
        Psz = [35, 35; 35, 35; 35, 35; 35, 35];
        mu = 1e0;
end

param = [];
param.Psz = Psz;
param.maxIter = 700;
param.tol = tol;
param.mu = mu; % Regularization parameter
param.plotFreq = 10; % Visualize iterations 
param.wName = 'db3';
param.wLevel = 3;
param.wMode = 'per';
param.x0 = x0;
xSz = [N,N,T];

% Iterate
[LMRLR_recn, LMRLR_iter, LMRLR_info] = LMRLRalgorithm(A, m, xSz, param);
LMRLR_info.param = param; % Save parameters

%% Look at outcome

f = gcf;
clf(f, "reset");
figure(f);
montage(LMRLR_recn,'DisplayRange', [])
title('Final LMRLR reconstruction')
drawnow

%% Run the LLR algorithm
% Set parameters
switch dataType
    case 'simulated'
        Psz = [8,8 ]; % Patch sizes (per scale)
        mu = 1e-1; % Regularization parameter
    case 'STEMPO'
        Psz = [7, 7];
        mu = 1e-1;
end

param = [];
param.maxIter = 600;
param.tol = tol;
param.mu = mu; % Regularization parameter
param.pSz = Psz;
param.plotFreq = 10; % Visualize iterations 
param.x0 = x0;
xSz = [N,N,T];

% Iterate
[LLR_recn, LLR_iter, LLR_info] = LLRalgorithm(A, m, xSz, param);
LLR_info.param = param; % Save parameters

%% Look at outcome

f = gcf;
clf(f, "reset");
figure(f);
montage(LLR_recn,'DisplayRange', [])
title('Final LLR reconstruction')
drawnow

%% Run the L+S algorithm
% Set parameters
switch dataType
    case 'simulated'
        % Regularization parameters
        muL = 0.35;
        muS = 0.02;
    case 'STEMPO'
        muL = 0.35;
        muS = 0.02;
end

param = [];
param.muL = muL;
param.muS = muS;
param.maxIter = 600;
param.tol = tol;
param.plotFreq = 10; % Visualize iterations 
param.wName = 'db3';
param.wLevel = 3;
param.wMode = 'per';
xSz = [N,N,T];

% Iterate
[L_recn, S_recn, LpS_iter, LpS_info] = LplusSalgorithm(A, m, xSz, param);
LpS_info.param = param;
LpS_recn = L_recn + S_recn;

%% Look at outcome

f = gcf;
clf(f, "reset");
figure(f);
montage(max(0, LpS_recn),'DisplayRange', [])
title('Final L+S reconstruction')
drawnow

figure;
tiledlayout(1, 2, "TileSpacing", "tight");
nexttile
montage(L_recn, 'DisplayRange', [])
title('Final L part')
colorbar
drawnow

nexttile
montage(S_recn, 'DisplayRange', [])
title('Final S part')
colorbar
drawnow

%% Print error metrics

% Pick one angle to represent each time step
avgAngle = angleArray(round(Nangles/2), :);

objInd = round(size(obj,3) * avgAngle / numberImages);
ref = obj(:,:,objInd);

figure;
montage(ref, 'DisplayRange', [])
title("Reference object")
colorbar

%%
%               l^2 relative error,      average SSIM,             average HaarPsi
LMRLR_errors = [relErr(LMRLR_recn, ref), avgSSIM(LMRLR_recn, ref), avgHPSI(LMRLR_recn, ref)];
LLR_errors   = [relErr(LLR_recn, ref),   avgSSIM(LLR_recn, ref),   avgHPSI(LLR_recn, ref)];
LpS_errors   = [relErr(LpS_recn, ref),   avgSSIM(LpS_recn, ref),   avgHPSI(LpS_recn, ref)];

disp(cat(1, [" ", "L2 rel.error", "avg. SSIM", "avg. HaarPSI"], [["LMRLR"; "LLR"; "L+S"], compose("%.3f", [LMRLR_errors; LLR_errors; LpS_errors])]));

%% Save stuff

maxVals = max([LMRLR_recn(:), LLR_recn(:), LpS_recn(:)]);

cm = colormap('jet');

f = figure(333);
f.Position = [100, 100, 512, 512];
f.Color = [1,1,1];
x = LpS_recn;
lims = [0, 0.9*max(maxVals)];
for t = 1:T
    ax=gca;
    ax.Position=[0 0 1 1];
    imagesc(x(:,:,t), lims);
    colormap(cm);
    axis equal;
    axis off;
    drawnow;
    print(sprintf("results/images/stempo/stempo_L+S_recn_t%d.png", t), '-dpng');
end
% colorbar; ax.FontSize = 14; ax.Position = [0.1300 0.1100 0.7750 0.8150];

%% Error metric functions

function r = relErr(x, ref)
% L^2 relative error
r = norm(x(:) - ref(:)) / norm(ref(:));
end

function s = avgSSIM(x, ref)
% Compute average SSIM between 3D objects
T = size(x,3);
st = zeros(1,T);

for t = 1:T
    st(t) = ssim(x(:,:,t), ref(:,:,t));
end
s = mean(st);
end

function h = avgHPSI(x, ref)
% Compute average HaarPSI between 3D objects
T = size(x,3);
ht = zeros(1,T);
% Normalize to [0, 255];
M = 255 / max([x(:); ref(:)]);
X = M*x;
REF = M*ref;

for t = 1:T
    ht(t) = HaarPSI(X(:,:,t), REF(:,:,t));
end
h = mean(ht);
end
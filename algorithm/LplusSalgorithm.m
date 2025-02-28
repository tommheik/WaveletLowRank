function [L, S, iter, info] = LplusSalgorithm(A, m, xSz, param)
%LPLUSSALGORITHM Computes Low-rank plus Sparse regularization
%algorithm given forward operator A, data m and additional parameters param
%
% INPUTS
%   A       Block diagoal forward operator
%   m       Data stack
%   xSz     Size of the reconstruction array: xSz = [N N T]
%   param   Optional parameters such as
%       maxIter     Maximum number of iterations. DEFAULT = 1000
%       wName       Wavelet type. DEFAULT = 'haar'
%       wLevel      Wavelet decomposition level. DEFAULT = 3
%       wMode       Convolution extension. DEFAULT = 'per'
%       tol         Tolerance for stopping criterion. DEFAULT = 1e-4
%       muL         Low-rank part regularization paramrter. DEFAULT = 1
%       muS         Sparse part regularization parameter. DEFAULT = 1
%       plotFreq    Plot intermediate reconstructions. DEFAULT = 0; (off)        
%
%%%%%%
%
% Example code for reconstructing the STEMPO phantom data using Low-rank +
% sparse decomposition (L+S) scheme. The aim is to separate the slowly
% changing low-rank background L and the more rapidly changing dynamic
% component S which should be made of only few nonzero terms, i.e. is
% sparse.
%
%%%%%%
%
% We wish to solve the minimization task:
%
% argmin_{A(L+S) = m} || L ||_* + mu*|| WS ||_1,                (1)
%
% where
% A is the forward operator, m is the sinogram and the reconstruction is
% split into the low-rank and sparse terms L and S respectively. L is
% constrained using the nuclear norm || ||_* which is the sum of singular
% values of the matrix (also denoted L) whose columns consists of the 
% different time steps: L = [L_1, ..., L_T].
%
% To further increase the sparsity of the dynamic component, we consider
% the wavelet transform WS_t of every time step of the dynamic component. 
%
% Mu is a positive regularization parameters and W is the 2D wavelet
% transform.
%
% In practice however we solve the regularization version of eq.(1) instead:
%
% argmin_{L, S} 1/2*|| A(L+S) - m ||_2^2 + mu_L || L ||_* + mu_S || WS ||_1,
%
% where we have separate regularization parameters mu_L and mu_s.
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
% Created 14.9.2022 - Last edited 23.9.2022
% Shared as part of the STEMPO dataset
% Tommi Heikkilä
% University of Helsinki
%
% Modified for comparison with other Low-rank methods
% 1.10.2024
% Tommi Heikkilä
% LUT University

% Check parameters, assign defaults if needed
if isfield(param,'maxIter'); maxIter = param.maxIter; else; maxIter = 1000; end
if isfield(param,'wName'); wname = param.wName; else; wname = 'haar'; end
if isfield(param,'wLevel'); level = param.wLevel; else; level = 3; end
if isfield(param,'wMode'); wmode = param.wMode; else; wmode = 'per'; fprintf('Using periodic convolutions!\n'); end
if isfield(param,'tol'); tol = param.tol; else; tol = 3; end
if isfield(param,'muL'); muL = param.muL; else; muL = 1; fprintf('Using default value for muL \n'); end
if isfield(param,'muS'); muS = param.muS; else; muS = 1; fprintf('Using default value for muS \n'); end
if isfield(param,'plotFreq'); plotfreq = param.plotFreq; else; plotfreq = 0; fprintf('Using default value for plotFreq\n'); end

N = xSz(1);
T = xSz(3);

m = m(:); % Drop to column vector

% Backproject
M = reshape(A'*m,[N*N,T]);
Lpre = M;

dwtmode(wmode, 'nodisp');
[~, wSz] = wavedec2(zeros(N,N), level, wname); % Wavelet decomp. size

S=zeros(N*N,T); 
iter=0;

% Store useful values during iteration
relChange = nan(1,maxIter);
dataFit = nan(1,maxIter);
nuclear = nan(1,maxIter);
l1Norm = nan(1,maxIter);

fprintf('----------\nBegin L+S!\n----------\n')
tic;

if plotfreq > 0
    f = figure;
end

%% Iterate
while iter < maxIter
    iter=iter+1;
    
    % low-rank update
    M0 = M;
    [Ut,St,Vt] = svd(M-S, "econ");
    St = diag(SoftThresh(diag(St), muL));
    L = Ut*St*Vt';

    % soft threshold M - Lpre on wavelet domain
    WS = SoftThresh(Wfwd(reshape(M - Lpre,xSz), level, wname), muS);
    S = reshape(Wadj(WS, wSz, wname),[N*N,T]);

    % data consistency
    LplusS = L + S;
    dif = A*LplusS(:) - m;
    M = max(0, LplusS - reshape(A'*dif,[N*N,T])); % Nonnegativity constraint is not exactly rigorous


    % L_{k-1} for the next iteration
    Lpre = L;
    
    relChange(iter) = norm(M(:)-M0(:))/norm(M0(:));
    dataFit(iter) = norm(dif);
    nuclear(iter) = sum(diag(St));
    l1Norm(iter) = norm(WS(:),1);
    
    if mod(iter,10) == 0
        fprintf('Iteration number %d reached \n', iter);
        fprintf('Relative change: %.5f \n', relChange(iter));
        fprintf('Cost function: %.2f \n', dataFit(iter)^2 + muL*nuclear(iter) + muS*l1Norm(iter));
        fprintf('----------\n')
    end
    
    if mod(iter,plotfreq) == 0
        figure(f)
        tiledlayout(1,3,"TileSpacing","tight");

        nexttile
        montage(reshape(L,xSz), 'DisplayRange', []);
        title(sprintf('L at iter: %d', iter));
        drawnow

        nexttile
        montage(reshape(S,xSz), 'DisplayRange', []);
        title(sprintf('S at iter: %d', iter));
        drawnow

        nexttile
        montage(max(0, reshape(S,xSz) + reshape(L, xSz)), 'DisplayRange', []);
        title(sprintf('L+S at iter: %d', iter));
        drawnow
    end
    
    % Stop once relative change is below tolerance
    if (relChange(iter) < tol) && (iter > 1) 
        fprintf('Stopping criterion reached after %d iterations \n',iter);
        break
    end
end
L = reshape(L,xSz);
S = reshape(S,xSz);

timeTot = toc;

fprintf('Total computational time: %.1f s, approximately %.2f s per iteration \n', timeTot, timeTot / iter);

if iter == maxIter
    fprintf('Maximum iteration count reached! Iteration stopped \n');
else
    % Cut stored arrays to correct length
    relChange = relChange(1:iter);
    dataFit = dataFit(1:iter);
    nuclear = nuclear(1:iter);
    l1Norm = l1Norm(1:iter);
end
info.relChange = relChange;
info.dataFit = dataFit;
info.nuclear = nuclear;
info.l1Norm = l1Norm;
info.functionalValues = dataFit.^2 + muL*nuclear + muS*l1Norm;
info.timeTot = timeTot;
end

function w = Wfwd(x, level, wname)
    %%% Perform 2D wavelet transform on every layer of x (3D array)
    
    T = size(x,3);
    [C,~] = wavedec2(x(:,:,1),level,wname);
    
    if T == 1 % Special case for 2D array
        w = C;
        return
    end
    w = zeros(T, length(C));
    w(1,:) = C;
    for t = 2:T
        [C,~] = wavedec2(x(:,:,t),level,wname);
        w(t,:) = C;
    end
end

function X = Wadj(w, wSz, wname)
    %%% Perform adjoint (inverse) of 2D wavelet transform on every layer of w
    
    T = size(w,1);
    x = waverec2(w(1,:),wSz,wname);
    
    if T == 1 % Special case for 1D array
        X = x;
        return
    end
    
    xSz = size(x);
    X = zeros([xSz, T]);
    X(:,:,1) = x;
    
    for t = 2:T
        X(:,:,t) = waverec2(w(t,:),wSz,wname);
    end
end
function [x, iter, info] = LLRalgorithm(A, m, xSz, param)
%LLRALGORITHM Computes the Local Low-Rank regularization algorithm given 
% forward operator A, data m and additional parameters param
%
% We wish to solve the fllowing minimization problem:
%   recn = argmin f(x) + mu*g(x) + h(x),
%   where f(x) = 1/2 ||Ax - m||^2, i.e. smooth data mismatch term,
%         g(x) = mu*||X_p||_*, i.e. nuclear norm of patched x and
%         h(x) = i_+(x), i.e. the indicator function on the nonnegative
%         orthant
% 
% Since patching is orthogonal, the prox of the nuclear norm is soft
% thresholding of the singular values of X, prox of i_+ is projection and
% we can solve it all with PDFP
%
% INPUTS
%   A       Block diagoal forward operator
%   m       Data stack
%   xSz     Size of the reconstruction array: xSz = [N N T]
%   param   Optional parameters such as
%       maxIter     Maximum number of iterations. DEFAULT = 1000
%       tol         Tolerance for stopping criterion. DEFAULT = 1e-4
%       mu          Regularization parameter. DEFAULT = 1;
%       patchSz     Size of the image patch. DEFAULT = 10; (10x10)
%       plotFreq    Plot intermediate reconstructions. DEFAULT = 0; (off)
%
% T H   2023

if nargin < 3
    param = struct([]);
end

% Check parameters, assign defaults if needed
if isfield(param,'maxIter'); maxIter = param.maxIter; else; maxIter = 1000; fprintf('Using default value for maxIter \n'); end
if isfield(param,'tol'); tol = param.tol; else; tol = 3; fprintf('Using default value for tol \n'); end
if isfield(param,'mu'); mu = param.mu; else; mu = 1; fprintf('Using default value for mu \n'); end
if isfield(param,'pSz'); pSz = param.pSz; else; pSz = 10; fprintf('Using default value for pSz \n'); end
if isfield(param,'plotFreq'); plotFreq = param.plotFreq; else; plotFreq = 0; fprintf('Using default value for plotFreq\n'); end
if isfield(param,'x0'); x = param.x0; else; x = zeros(xSz, class(m)); fprintf('Using default value for x0\n'); end

m = m(:); % Drop to column vector
vec = @(x) x(:);

% Initialize
v = array2patch(x, pSz);
bp = -A'*m; % This is needed later

iter = 0;

% PDFP parameters
lambda = 0.99; % < 1/lambda_max(P P^T), P = patching operator
gamma = 1; % < 2/Lipschitz constant of the gradient of f

% Store useful values during iteration
relChange = nan(1,maxIter);
dataFit = nan(1,maxIter);
nuclear = nan(1,maxIter);

tic;
fprintf('----------\nBegin LLR!\n----------\n')

if plotFreq > 0
    f = figure;
end

%% Iterate
while iter < maxIter
    iter=iter+1;
    
    % % low-rank update
    % M0 = b; % Column vector
    % P = array2patch(reshape(b, xSz), pSz); % Break 2D images into patches
    % [Plr, nuclearList, ~] = SVThreshold(P,mu); % Threshold singular values
    % b = patch2array(Plr, pSz, xSz); % Combine back to 2D + time array, 3D array

    xOld = x(:); % x is 3D array, xOld is column vector

    % PDFP steps
    y = max(0, x - gamma*reshape(bp,xSz) - lambda*patch2array(v, pSz, xSz)); % 3D array
    By = array2patch(y, pSz);
    [prox2, nuclearList] = SVThreshold(v + By, mu); % This is not the correct value for the nuclear norm
    v = v + By - prox2;
    x = max(0, xOld - gamma*bp - lambda*vec(patch2array(v, pSz, xSz))); % Column vector

    % data consistency
    dif = A*x - m; % Column vector
    bp = A'*dif;

    
    relChange(iter) = norm(x - xOld)/norm(xOld);
    dataFit(iter) = norm(dif);
    nuclear(iter) = sum(nuclearList); % Combine all the nuclear norm of each block

    x = reshape(x, xSz);
    
    if mod(iter,10) == 0
        fprintf('Iteration number %d reached \n', iter);
        fprintf('Relative change: %.5f \n', relChange(iter));
        fprintf('Cost function: %.2f \n', dataFit(iter)^2 + mu*nuclear(iter));
        fprintf('----------\n')
    end
    
    if mod(iter,plotFreq) == 0
        figure(f)
        montage(x, 'DisplayRange', []);
        title(sprintf('LLR Reconstruction at iter: %d', iter));
        drawnow
    end
    
    % Stop once relative change is below tolerance
    if relChange(iter) < tol
        fprintf('Stopping criterion reached after %d iterations \n',iter);
        break
    end
end

timeTot = toc;

fprintf('Total computational time: %.1f s, approximately %.2f s per iteration \n', timeTot, timeTot / iter);

if iter == maxIter
    fprintf('Maximum iteration count reached! Iteration stopped \n');
else
    % Cut stored arrays to correct length
    relChange = relChange(1:iter);
    dataFit = dataFit(1:iter);
    nuclear = nuclear(1:iter);
end
info.relChange = relChange;
info.dataFit = dataFit;
info.nuclear = nuclear;
info.functionalValues = dataFit.^2 + mu*nuclear;
info.timeTot = timeTot;
end


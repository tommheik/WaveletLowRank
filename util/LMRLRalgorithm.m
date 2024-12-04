function [x, iter, info] = LMRLRalgorithm(A, m, xSz, param)
%LMRLRALGORITHM Computes the Local MultiResolution Low-Rank regularization
%algorithm given forward operator A, data m and additional parameters param
%
% INPUTS
%   A       Block diagoal forward operator
%   m       Data stack
%   xSz     Size of the reconstruction array: xSz = [N N T]
%   param   Optional parameters such as
%       Psz         Patch sizes for every decomposition level.
%       maxIter     Maximum number of iterations. DEFAULT = 1000
%       wName       Wavelet type. DEFAULT = 'haar'
%       wLevel      Wavelet decomposition level. DEFAULT = 3
%       wMode       Convolution extension. DEFAULT = 'per'
%       tol         Tolerance for stopping criterion. DEFAULT = 1e-4
%       mu          Regularization paramter. DEFAULT = 1
%       plotFreq    Plot intermediate reconstructions. DEFAULT = 0; (off)        
%
% T H   2023

if nargin < 3
    param = struct([]);
end

% Check parameters, assign defaults if needed
if isfield(param,'maxIter'); maxIter = param.maxIter; else; maxIter = 1000; end
if isfield(param,'wName'); wName = param.wName; else; wName = 'haar'; end
if isfield(param,'wLevel'); wLevel = param.wLevel; else; wLevel = 3; end
if isfield(param,'wMode'); wMode = param.wMode; else; wMode = 'per'; fprintf('Using periodic convolutions!\n'); end
if isfield(param,'tol'); tol = param.tol; else; tol = 3; end
if isfield(param,'mu'); mu = param.mu; else; mu = 1; fprintf('Using default value for mu \n'); end
if isfield(param,'plotFreq'); plotFreq = param.plotFreq; else; plotFreq = 0; fprintf('Using default value for plotFreq\n'); end
if isfield(param,'x0'); x = param.x0; else; x = zeros(xSz, class(m)); fprintf('Using default value for x0\n'); end

T = xSz(3);


dwtmode(wMode,'nodisp');
% Get wavelet coefficient array size
[~, Csz] = wavedec2(x(:,:,1),wLevel,wName);
Csz = flipud(Csz(1:end-1,:));
W.Csz = Csz;

% Patch size
Psz = param.Psz;
fprintf("=== Subband and patch sizes per level ===\n")
disp([' Coeff. ', 'sizes ', ' Patch ', 'sizes '])
disp([Csz, Psz])
W.level = wLevel;
W.name = wName;
W.mode = wMode;

m = m(:); % Drop to column vector
vec = @(x) x(:);

% InitializeBx = array1
v = zeros([sum(Csz),T], class(m));
invBv = zeros(xSz, class(m)); % B^T(v), but v is initialized as zeros
bp = A'*(-m); % This is needed later

iter = 0;

% PDFP parameters
lambda = 0.99; % < 1/lambda_max(P P^T), P = patching operator
gamma = 1; % < 2/Lipschitz constant of the gradient of f

% Store useful values during iteration
relChange = nan(1,maxIter);
dataFit = nan(1,maxIter);
nuclear = nan(1,maxIter);
objFun = nan(1,maxIter);

tic;
fprintf('----------\nBegin LMRLR!\n----------\n')

if plotFreq > 0
    f1 = figure;
    f2 = figure;
end

%% Iterate
while iter < maxIter
    iter=iter+1;
    

    xOld = x(:); % x is 3D array, xOld is column vector

    % PDFP steps
    y = max(0, x - gamma*reshape(bp,xSz) - lambda*invBv); % 3D array
    By = array2Wpatch(y, W, 0, Psz); % No thresholding
    [prox2, nn] = array2Wpatch(invBv + y, W, mu, Psz); % This is not the correct value for the nuclear norm
    v = v + By - prox2;
    invBv = Wpatch2array(v, W, xSz);
    x = max(0, xOld - gamma*bp - lambda*vec(invBv)); % Column vector

    % data consistency
    dif = A*x - m; % Column vector
    bp = A'*dif;

    
    relChange(iter) = norm(x - xOld)/norm(xOld);
    dataFit(iter) = norm(dif);
    nuclear(iter) = nn;
    objFun(iter) = dataFit(iter)^2 + mu*nuclear(iter);

    x = reshape(x, xSz);
    
    if mod(iter,10) == 0
        fprintf('Iteration number %d reached \n', iter);
        fprintf('Relative change: %.5f \n', relChange(iter));
        fprintf('Cost function: %.2f \n', objFun(iter));
        fprintf('----------\n')
    end
    
    if mod(iter,plotFreq) == 0
        figure(f1)
        montage(x, 'DisplayRange', []);
        title(sprintf('LMRLR Reconstruction at iter: %d', iter));
        drawnow

        figure(f2)
        t = mod(round(iter/plotFreq),xSz(3)) + 1;
        imagesc(prox2(:,:,t));
        title(sprintf('LMRLR: Wavelet array (t=%d) at iter: %d', t, iter));
        axis equal
        axis off
        drawnow
    end
    
    % Stop once relative change is below tolerance
    if relChange(iter) < tol
        fprintf('Stopping criterion reached after %d iterations \n',iter);
        break
    end
    % Stop if something is wrong
    if isnan(objFun(iter)) || isinf(objFun(iter))
        fprintf('Something wrong! Stopping after %d iterations \n',iter);
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

W.Psz = Psz;
info.MRSystem = W;
info.relChange = relChange;
info.dataFit = dataFit;
info.nuclear = nuclear;
info.functionalValues = dataFit.^2 + mu*nuclear;
info.timeTot = timeTot;
end


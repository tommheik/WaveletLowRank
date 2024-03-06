function [Y, nuclear, sparsity] = SVThreshold(X, mu)
%SVThreshold Singular Value SoftThresholding
%   Given a collection of matrices X = [x_1, x_2, ..., x_N], soft threshold
%   the singular values of each matrix x_n based on parameter mu and return
%   the collection of thresholded matrices Y.
%   nuclear is the nuclear norm or sum of singular values of Y
%   'sparsity' measures the number of nonzero singular values left after 
%   thresholding.
%
% Uses the MTIMESX routine by James Tursa if available
% James Tursa (2023). MTIMESX - Fast Matrix Multiply with Multi-Dimensional Support 
% (https://www.mathworks.com/matlabcentral/fileexchange/25977-mtimesx-fast-matrix-multiply-with-multi-dimensional-support), 
% MATLAB Central File Exchange. Retrieved June 16, 2023. 
%
% T H   2023

[U, S, V] = pagesvd(X, "econ", "vector");
Ssz = size(S);

SmuS = SoftThresh(S,mu);

S = repmat(S,1,Ssz(1)); % Fatten S along second dimension

sparsity = zeros(Ssz(2:end));
nuclear = sum(squeeze(SmuS),1); % Sum of singular values for each S_mu
for i = 1:prod(Ssz(2:end))
    sparsity(i) = nnz(SmuS(:,i)) / numel(SmuS(:,i));
    S(:,:,i) = diag(SmuS(:,i)); % Expand to matrix
end
if exist('mtimesx', 'file')
    % (supposedly) Fast matrix multiplications of nD arrays
    X = mtimesx(S,V,'t');
    Y = mtimesx(U,X);
else
    % Normal matrix multiplications
    Y = zeros(size(X),class(X));
    for i = 1:prod(Ssz(3:end))
        X = S(:,:,i)*V(:,:,i)';
        Y(:,:,i) = U(:,:,i)*X;
    end
end
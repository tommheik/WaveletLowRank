function P = array2patch(A,pSz)
% Function to patch any nD array A into pSz sized 2D patches
% Example
% Let A be 3D array [A_1, A_2, ..., A_T], where each A_t is made of blocks
% of size pSz = p_1 x p_2
%
%       |b_11 b_12 ... b_1n|
%       |b_21 b_22 ... b_2n|
% A_t = | .    .        .  |, then P(:,t,:) = [b_11(:), b_12(:), ..., b_mn(:)]
%       | .    .        .  |
%       |b_m1 b_m2 ... b_mn|
%
% Note: all patches are dropped into column vectors for SVD later
%
% T H   2023

if ndims(A) ~= 3
    error('A should be 3D array!')
end
aSz = size(A);
T = aSz(3);

if length(pSz) == 1
    pSz = pSz*[1,1]; % Extend to 2 values
end

% Check if patch size is compatible
if (mod(aSz(1),pSz(1)) ~= 0); error('A has %d rows which is not divisible by %d!',aSz(1),pSz(1)); end
if (mod(aSz(2),pSz(2)) ~= 0); error('A has %d columns which is not divisible by %d!',aSz(2),pSz(2)); end

pLen = pSz(1)*pSz(2);
pNum = aSz(1:2)./pSz; % Number of patches

P = zeros([pLen,T,pNum(1)*pNum(2)],class(A)); % Reorganize P into b(:) x T x pNum array

% Loop through rows and columns
rInd = 0;
for row = 1:pNum(1)
    cInd = 0;
    for col = 1:pNum(2)
        blk = reshape(A(rInd+1:rInd+pSz(1),cInd+1:cInd+pSz(2),:),[pLen,T]);
        P(:,:,(row-1)*pNum(2)+col) = blk;
        cInd = cInd + pSz(2);
    end
    rInd = rInd + pSz(1);
end
end
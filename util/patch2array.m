function A = patch2array(P, pSz, aSz)
% Function to combine pSz sized patches back into 3D array A of size aSz
%
% T H   2023

if length(pSz) == 1
    pSz = pSz*[1,1]; % Extend to 2 values
end

% Check if patch size is compatible
if (mod(aSz(1),pSz(1)) ~= 0); error('A has %d rows which is not divisible by %d!',aSz(1),pSz(1)); end
if (mod(aSz(2),pSz(2)) ~= 0); error('A has %d columns which is not divisible by %d!',aSz(2),pSz(2)); end

pNum = aSz(1:2)./pSz; % Number of patches

T = aSz(3);
A = zeros(aSz,class(P));

% Loop through rows and columns
rInd = 0;
for row = 1:pNum(1)
    cInd = 0;
    for col = 1:pNum(2)
        blk = reshape(P(:,:,(row-1)*pNum(2)+col),[pSz,T]);
        A(rInd+1:rInd+pSz(1),cInd+1:cInd+pSz(2),:) = blk;
        cInd = cInd + pSz(2);
    end
    rInd = rInd + pSz(1);
end
end
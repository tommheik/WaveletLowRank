function [Wpatch, nuclear] = array2Wpatch(A,W,mu,Psz)
% Function to patch any 3D array A into 2D wavelet coefficient patches. If
% threshold value 'mu' is positive, then each wavelet subband gets it
% singular values soft thresholded
%
% Input
% A     3D input array
% W     Struct carrying information about the wavelet: 'wname', 'level',
%       'mode' (DEFAULT = 'per')
% mu    Soft thresholding parameter for singular values. For mu = 0 SVD is
%       NOT computed and P is just the wavelet transform of A
%
% T H   2023

if ndims(A) ~= 3
    error('A should be 3D array!')
end

Asz = size(A);
T = Asz(3);
if mu < 0
    error('mu should be nonnegative!')
end

if ~isfield(W,'mode')
    % Default extension
    W.mode = 'per';
end

Aclass = class(A);
if nargin < 4
    % Subband size missing
    Psz = repmat(Asz(1:2),W.level,1);
    for l = 1:W.level
        Psz(l:end,:) = ceil(Psz(l:end,:)/2); % Iteratively get size of every subband
    end
    Psz = [Psz;Psz(end,:)]; % Approximation coefficients
    % This Psz only works with 'per'iodice convolution
    W.mode = 'per';
elseif size(Psz,1) == W.level + 2
    % For consistency: Psz contains input size
    Psz = Psz(2:end,:);
elseif size(Psz,1) == W.level
    % For consistency: Psz missing approximation coefficient size
    Psz = [Psz;Psz(end,:)]; % Approximation coefficients
end
Wsz = sum(Psz,1); % Total wavelet grid size
Cind = Wsz - cumsum(Psz,1); % Cumulative sum of sizes

switch mu
    case 0 % Just do 2D dwt
        nuclear = nan;
        Wpatch = zeros([Wsz, T], class(A));
        P = zeros(Wsz, class(A));
        for t = 1:T
            a = A(:,:,t);
            for l = 1:W.level
                [a,v,h,d] = dwt2(a,W.name,'mode',W.mode); % Different order compared to wavedec2
                R = Cind(l,1); C = Cind(l,2); % Cumulative index
                row = 1:Psz(l,1); col = 1:Psz(l,2);
                P(R+row,col) = v;
                P(row,C+col) = h;
                P(R+row,C+col) = d;
            end
            P(row,col) = a;
            Wpatch(:,:,t) = P;
        end

    otherwise % We can be clever and do every time step per level
        nuclear = 0;
        Wpatch = zeros([Wsz, T], class(A));
        for l = 1:W.level
            nSub = 3 + floor(l/W.level); % Number of subbands
            P = zeros([prod(Psz(l,:)),T,nSub],Aclass);
            Atemp = zeros([Psz(l,:),T], Aclass);
            for t = 1:T
                [a,v,h,d] = dwt2(A(:,:,t),W.name,'mode',W.mode);
                P(:,t,1) = v(:); P(:,t,2) = h(:); P(:,t,3) = d(:);
                Atemp(:,:,t) = a;
            end
            if l < W.level
                A = Atemp;
            else % Last level is different
                P(:,:,4) = reshape(Atemp,[],T);
            end
            % Compute SVD on P
            [Y, nn] = SVThreshold(P, mu);
            nuclear = nuclear + sum(nn); % Update nuclear norm
            
            % Store wavelet coefficients back to grid
            R = Cind(l,1); C = Cind(l,2); % Cumulative index
            row = 1:Psz(l,1); col = 1:Psz(l,2);
            Wpatch(R+row,col,:) = reshape(Y(:,:,1),[Psz(l,:),T]); % v
            Wpatch(row,C+col,:) = reshape(Y(:,:,2),[Psz(l,:),T]); % h
            Wpatch(R+row,C+col,:) = reshape(Y(:,:,3),[Psz(l,:),T]); % d
        end
        Wpatch(row,col,:) = reshape(Y(:,:,4),[Psz(l,:),T]); % a
end

end
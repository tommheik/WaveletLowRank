function A = Wpatch2array(Wpatch,W,Asz,Psz)
% Function to reconstruct a 3D array A from the 2D wavelet patches Wpatch
% using the inverse wavelet transform defined using W.
% sz can be either the size of the wavelet subbands (Psz) or A (Asz).
%
% Input
% Wpatch    3D input array of wavelet coefficients
% W     Struct carrying information about the wavelet: 'wname', 'level',
%       'mode' (DEFAULT  = 'per')
% Asz   The desired size of A, Psz can be computed from this
% Psz   The size of each wavelet subband since there may be filler
%
% T H   2023

Wsz = size(Wpatch);
T = Wsz(3);

if ~isfield(W,'mode')
    % Default extension
    W.mode = 'per';
end

Asz = Asz(:)'; % Flatten
Asz = Asz(1:2);
if nargin < 4 % Psz missing
    Psz = repmat(Asz,W.level,1);
    for l = 1:W.level
        Psz(l:end,:) = ceil(Psz(l:end,:)/2); % Iteratively get size of every subband
    end
    Psz = [Psz;Psz(end,:)]; % Approximation coefficients
    % This Psz only works with 'per'iodice convolution
    W.mode = 'per';
end
if size(Psz,1) < W.level +2
    Psz = [Asz; Psz]; % Add one more layer on top
end
Asz = [Asz, T];
Cind = Wsz(1:2) - cumsum(Psz(2:end,:),1); % Cumulative sum of sizes

A = zeros(Asz, class(Wpatch));
for t = 1:T
    P = Wpatch(:,:,t);
    a = P(1:Psz(end,1),1:Psz(end,2));
    for l = W.level+1:-1:2
        R = Cind(l-1,1); C = Cind(l-1,2); % Cumulative index
        row = 1:Psz(l,1); col = 1:Psz(l,2);
        v = P(R+row,col);
        h = P(row,C+col);
        d = P(R+row,C+col);
        a = idwt2(a,v,h,d,W.name,'mode',W.mode,Psz(l-1,:)); % New size is from next level
    end
    A(:,:,t) = a;
end


end
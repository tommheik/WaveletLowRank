function X = Wpatch2array(Wpatch,W,Xsz)
% Function to reconstruct a 3D array X from the 2D wavelet patches Wpatch
% using the inverse wavelet transform defined using W.
% sz can be either the size of the wavelet subbands (Psz) or X (Xsz).
%
% Input
% Wpatch    3D input array of wavelet coefficients
% W     Struct carrying information about the wavelet: 'wname', 'level',
%       'mode' (DEFAULT  = 'per') and 'Csz' (wavelet subband sizes)
% Xsz   The desired size of X
%
% T H   2023 (edited 2024)

Wsz = size(Wpatch);
T = Wsz(3);

if ~isfield(W,'mode')
    % Default extension
    W.mode = 'per';
end

Csz = W.Csz; % Coefficient sizes from wavedec2 [from finest to coarsest]

Xsz = Xsz(:)'; % Flatten
Xsz = Xsz(1:2);

if size(Csz, 1) == W.level + 1
    % For consistency: Psz contains input size
    Csz = [Xsz; Csz];
elseif size(Csz, 1) == W.level
    % For consistency: Psz missing approximation coefficient size
    Csz = [Xsz; Csz; Csz(end,:)]; % Approximation coefficients
end

Xsz = [Xsz, T];
Cind = Wsz(1:2) - cumsum(Csz(2:end,:),1); % Cumulative sum of sizes

X = zeros(Xsz, class(Wpatch));
for t = 1:T
    P = Wpatch(:,:,t);
    a = P(1:Csz(end,1),1:Csz(end,2));
    for j = W.level+1:-1:2
        R = Cind(j-1,1); C = Cind(j-1,2); % Cumulative index
        row = 1:Csz(j,1); col = 1:Csz(j,2);
        v = P(R+row,col);
        h = P(row,C+col);
        d = P(R+row,C+col);
        a = idwt2(a,v,h,d,W.name,'mode',W.mode,Csz(j-1,:)); % New size is from next level
    end
    X(:,:,t) = a;
end


end
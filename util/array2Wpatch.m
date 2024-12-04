function [Wpatch, nuclear] = array2Wpatch(X, W, mu, Psz)
% Function to patch any 3D array A into 2D wavelet coefficient patches. If
% threshold value 'mu' is positive, then each wavelet subband gets it
% singular values soft thresholded
%
% Input
% X     3D input array
% W     Struct carrying information about the wavelet: 'wname', 'level',
%       'mode' (DEFAULT = 'per') and 'Csz'
% mu    Soft thresholding parameter for singular values. For mu = 0 SVD is
%       NOT computed and P is just the wavelet transform of A
% Psz   Patch size for each decomposition level
%
% T H   2023 (edited 2024)

vec = @(x) x(:);

if ndims(X) ~= 3
    error('A should be 3D array!')
end

Xsz = size(X);
T = Xsz(3);
if mu < 0
    error('mu should be nonnegative!')
end

if ~isfield(W,'mode')
    % Default extension
    W.mode = 'per';
end

Csz = uint16(W.Csz); % Coefficient sizes from wavedec2 [from finest to coarsest]

Xclass = class(X);
if nargin < 4
    % Subband size missing
    Psz = uint16(Csz);
    nP = ones(W.level + 1,2, 'uint16'); % Number of patches
elseif size(Psz, 1) == W.level + 2
    % For consistency: Psz contains input size
    Psz = uint16(Psz(2:end, :));
    nP = uint16(Csz ./ Psz); % Number of patches
elseif size(Psz, 1) == W.level
    % For consistency: Psz missing approximation coefficient size
    Psz = uint16([Psz; Psz(end,:)]); % Approximation coefficients
    nP = uint16(Csz ./ Psz); % Number of patches
else
    Psz = uint16(Psz);
    nP = uint16(Csz ./ Psz); % Number of patches
end

Pmismatch = (nP .* Psz ~= Csz);
if any(Pmismatch)
    [~, i] = max(Pmismatch);
    i = min(i);
    error("Incompatible patch size for level %i: \n %i x %i subband, but %i x %i patches of size %i x %i", ...
        i, Csz(i,1), Csz(i,2), nP(i,1), nP(i,2), Psz(i,1), Psz(i,2));
end

Wsz = uint16(sum(Csz, 1)); % Total wavelet grid size
Cind = Wsz - cumsum(Csz,1); % Cumulative sum of sizes

switch mu
    case 0 % Just do 2D dwt, no patching needed
        nuclear = nan;
        Wpatch = zeros([Wsz, T], class(X));
        P = zeros(Wsz, class(X));
        for t = 1:T
            a = X(:,:,t);
            for j = 1:W.level
                [a,v,h,d] = dwt2(a,W.name,'mode',W.mode); % Different order compared to wavedec2
                R = Cind(j,1); C = Cind(j,2); % Cumulative index
                row = 1:Csz(j,1); col = 1:Csz(j,2);
                P(R+row,col) = v;
                P(row,C+col) = h;
                P(R+row,C+col) = d;
            end
            P(row,col) = a;
            Wpatch(:,:,t) = P;
        end

    otherwise % We can be clever and do every time step per level
        nuclear = 0;
        Wpatch = zeros([Wsz, T], class(X));
        for j = 1:W.level
            Np = [prod(nP(j,:)), prod(Psz(j,:))]; % product sizes: no. patches, patch size
            V = zeros([Np(2), T, Np(1)], Xclass);
            H = zeros([Np(2), T, Np(1)], Xclass);
            D = zeros([Np(2), T, Np(1)], Xclass);
            if j < W.level
                A = zeros([Csz(j,:), T], Xclass);
            else
                A = zeros([prod(Psz(end,:)), T, prod(nP(end,:))], Xclass);
                % How A is stored
                if prod(nP(end,:)) == 1 % Only one patch
                    aInd = 1:prod(Psz(end,:));
                else % Multiple patches
                    q1 = Psz(j+1,1); q2 = Psz(j+1,2);
                    m = Csz(j+1,1); % Rows in coeff. subband
                    pic = vec((1:q1)' + (0:q2-1)*m); % Patch Index Column for first patch
                    pgr = vec(q1*(0:nP(j+1,1)-1)' + q2*m*(0:nP(j+1,2)-1));
                    aInd = pic + pgr'; % Patch indicies. This can skip time information
                end
            end

            R = Cind(j,1); C = Cind(j,2); % Cumulative index
            row = 1:Csz(j,1); col = 1:Csz(j,2);

            if Np(1) == 1 % Only one patch
                for t = 1:T
                    [a,v,h,d] = dwt2(X(:,:,t), W.name, 'mode', W.mode);
                    V(:,t) = v(:); H(:,t) = h(:); D(:,t) = d(:);
                    if j < W.level
                        A(:,:,t) = a;
                    else % Final appr. coeffs. are patched too
                        A(:,t,:) = a(aInd);
                    end
                end
                X = A;

                % Compute SVD on patched coefficient arrays, the output is of
                % size p1*p2 x T
                [A, nn] = SVThreshold(V, mu);
                nuclear = nuclear + sum(nn); % Update nuclear norm
                [V, nn] = SVThreshold(H, mu);
                nuclear = nuclear + sum(nn); % Update nuclear norm
                [H, nn] = SVThreshold(D, mu);
                nuclear = nuclear + sum(nn); % Update nuclear norm

                % Store wavelet coefficients back to grid
                Wpatch(R+row, col, :) = reshape(A,[Csz(j,:),T]); % v
                Wpatch(row, C+col, :) = reshape(V,[Csz(j,:),T]); % h
                Wpatch(R+row, C+col, :) = reshape(H,[Csz(j,:),T]); % d

            else % Multiple patches for detail coefficients
                p1 = Psz(j,1); p2 = Psz(j,2);
                m = Csz(j,1); % Rows in coeff. subband
                pic = vec((1:p1)' + (0:p2-1)*m); % Patch Index Column for first patch
                pgr = vec(p1*(0:nP(j,1)-1)' + p2*m*(0:nP(j,2)-1));
                pInd = pic + pgr'; % Patch indicies. This can skip time information

                if j == W.level
                    if any(Psz(j+1,:) ~= Psz(j,:)) % Unique patch size for approximation coefficients
                        q1 = Psz(j+1,1); q2 = Psz(j+1,2);
                        m = Csz(j+1,1); % Rows in coeff. subband
                        pic = vec((1:q1)' + (0:q2-1)*m); % Patch Index Column for first patch
                        pgr = vec(q1*(0:nP(j+1,1)-1)' + q2*m*(0:nP(j+1,2)-1));
                        aInd = pic + pgr'; % Patch indicies. This can skip time information
                    else
                        aInd = pInd;
                    end
                else
                end

                for t = 1:T
                    [a,v,h,d] = dwt2(X(:,:,t), W.name, 'mode', W.mode);
                    V(:,t,:) = v(pInd);
                    H(:,t,:) = h(pInd);
                    D(:,t,:) = d(pInd);
                    if j < W.level
                        A(:,:,t) = a;
                    else % Final appr. coeffs. are patched too
                        A(:,t,:) = a(aInd);
                    end
                end
                X = A;
                % Reverting indicies, this one can NOT skip time information!
                np1 = uint32(nP(j,1)); np2 = uint32(nP(j,2)); % Number of patches
                sic = vec(uint32(1:p1)' + (0:np1-1)*uint32(Np(2)*T)) + uint32((0:p2-1)*p1); % Indicies for Some Inverted Columns (for t=1)
                aic = reshape(sic(:) + (0:np2-1)*uint32(Np(2)*T*np1), Csz(j,:)); % Indicies for All Inverted Columns (for t=1)
                PInd = aic + uint32(reshape((0:T-1)*Np(2), 1, 1, T));

                % Compute SVD on patched coefficient arrays, the output is of
                % size p1*p2 x T x Np1*Np2
                [A, nn] = SVThreshold(V, mu); % Overwrite A
                nuclear = nuclear + sum(nn); % Update nuclear norm
                Wpatch(R+row, col, :) = A(PInd);

                [A, nn] = SVThreshold(H, mu);
                nuclear = nuclear + sum(nn); % Update nuclear norm
                Wpatch(row, C+col, :) = A(PInd);

                [A, nn] = SVThreshold(D, mu);
                nuclear = nuclear + sum(nn); % Update nuclear norm
                Wpatch(R+row, C+col, :) = A(PInd);

            end % if one patch or multiple
        end % for scale j

        % Approximation coefficients
        row = 1:Csz(j+1,1); col = 1:Csz(j+1,2);
        if prod(nP(j+1,:)) == 1 % Approx. coeff. use one patch only
            [A, nn] = SVThreshold(X, mu);
            nuclear = nuclear + sum(nn); % Update nuclear norm
            Wpatch(row, col, :) = reshape(A,[Csz(j,:),T]); % a

        else % Approx. coeff. use multiple patches
            pLen = prod(Psz(j+1,:));
            % Reverting indicies, this one can NOT skip time information!
            nq1 = uint32(nP(j+1,1)); nq2 = uint32(nP(j+1,2)); % Number of patches
            sic = vec(uint32(1:q1)' + (0:nq1-1)*uint32(pLen*T)) + uint32((0:q2-1)*q1); % Indicies for Some Inverted Columns (for t=1)
            aic = reshape(sic(:) + (0:nq2-1)*uint32(pLen*T*nq1), Csz(j,:)); % Indicies for All Inverted Columns (for t=1)
            AInd = aic + uint32(reshape((0:T-1)*pLen, 1, 1, T));

            [A, nn] = SVThreshold(X, mu);
            nuclear = nuclear + sum(nn); % Update nuclear norm
            Wpatch(row, col, :) = A(AInd);
        end
        
end % switch mu

end
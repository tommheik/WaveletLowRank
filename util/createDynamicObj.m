function obj = createDynamicObj(Nx, T, varargin)
% CREATEDYNAMICOBJ
% Creates a simple dynamic "animation" made of T consecutive Nx x Nx images
% The dynamics contain translations, deformations and anything in between.
% The final object is NOT necessarily normalized!
% 
% INPUTS
%   Nx      Spatial resolution (side length)
%   T       Time resolution
%
%   OPTIONAL INPUTS (keyword, value pairs in any order)
%   translations    List of translated objects ADDED. Negative values
%                   correspond to mirrored movement. Too many translations
%                   may lead to non-constant maximum brightness!
%                   Example: 'translation', [1, -1, 3];
%   deformation     Strength of AFFINE deformation applied to the STATIC
%                   structure. Negative values mirror the direction.
%                   Example: 'deformation', 1.3;
%   rotation        Total rotation angle (in DEGREES) applied to the static
%                   object. Example: 'roration', 15;
%   params          Parameter structure for ALL values. Crazy complicated
%
% T. Heikkilä, LUT University 2024

% Extra arguments and default values
translations = [];
deformation = 0;
rotation = 0;
params = struct([]);
for n = 1:2:length(varargin)
    narg = lower(varargin{n});
    switch narg
        case 'translations'
            translations = unique(varargin{n+1});
        case 'deformation'
            deformation = varargin{n+1};
        case 'rotation'
            rotation = varargin{n+1};
        case 'params'
            params = varargin{n+1};
        otherwise
            error("Unknown input argument '%s' given", narg);
    end
end

P = fillDefaultParameters(params);

background = zeros(Nx, Nx);

% Rotated object
rotLayer = zeros(Nx, Nx);
rotAngs = linspace(-rotation,rotation,T);

%% Rectangular block
k1 = round(P.blockSz(1) * Nx);
k2 = round(P.blockSz(2) * Nx);
brd = max(1, round(P.blockBrd * Nx)); % Border width
background(k1+1-brd:Nx-4*k1+brd,k2+1-brd:Nx-k2+brd) = P.blockBrdVal; % Background attenuation values
background(k1+1:Nx-4*k1,k2+1:Nx-k2) = P.blockVal;

[X, Y] = meshgrid(linspace(-1,1,Nx));
sigma  = P.sigma;
% Minor gradient to background
background = background .* (0.1 + 0.9*exp(-sigma*(X.^2 + Y.^2)));

%% Multiple intricate circles
radii = round(P.circRadii*Nx);
ys = round(P.circys*Nx);
xs = round(P.circxs*Nx);
ps = P.circps;
vals = P.circVals;

for i = 1:length(radii)
    % Add circles to top of image
    r = radii(i);
    x = xs(i);
    y = ys(i);
    p = ps(i);
    v = vals(i);
    if i == 2
        % Second circle gets holes
        % Second hole might be rotated
        sx = round(P.circHoleDist*2*r);
        mask = 1 - circshift(roundedRectangle(2*r, 2*r, 2, 1, P.circHoleSzs(1)), sx) - circshift(roundedRectangle(2*r, 2*r, 2, 1, P.circHoleSzs(2)), -sx);
        v = v * mask;
    end
    grad = P.circGradVal^i * linspace(0.9, 1.1, 2*r);
    if mod(i,2) == 0
        grad = grad';
    end
    circ = v .* roundedRectangle(2*r, 2*r, p, grad);
    if i == 1
        % First circle has wavy texture
        z = linspace(-1,1,2*r);
        waves = (1 - P.waveParams(1)) + P.waveParams(1)*(sin(P.waveParams(2)*pi*z).*sin(P.waveParams(3)*pi*z)');
        circ = circ .* waves;
    end
    if (i == 2) && (rotation ~= 0)
        % Save these for later
        circ2 = circ;
        c2x = x-r+1:x+r;
        c2y = y-r+1:y+r;
    else
        background(x-r+1:x+r, y-r+1:y+r) = background(x-r+1:x+r, y-r+1:y+r) + circ;
    end
        
end

% Normalization constant for background
bgMax = max(background(:));

%% Setup dynamics
obj = zeros(Nx, Nx, T);

n = round(P.m1sz(1)*Nx);
m = round(P.m1sz(2)*Nx);

grad = linspace(0.9, 1.1, 2*m);

r = round(P.m2sz*Nx);
grad2 = linspace(0.4, 0.6, 2*r)';

lim = P.movementLimit; % Limit of progress for movement: [lim, 1 - lim]

emptybg = zeros(Nx, Nx);
objCell = cell(2 + length(translations), 1);
objCell{1} = background;
objCell{2} = rotLayer;
objCell(3:end) = {emptybg};

for t = 1:T
    % Rotated object
    if exist("circ2", "var")
        rotLayer(c2x, c2y) = imrotate(circ2, rotAngs(t), 'bilinear', 'crop');
        objCell{2} = rotLayer;
    end

    prog = lim + (1 - 2*lim)*(t/T); % Progress
    % rectangle 1
    scale = 0.8 + 0.3*sin(0.8*pi*prog);
    hole = 1 - roundedRectangle(2*n, 2*m, 2, 1, 0.3*scale);
    rec = roundedRectangle(2*n, 2*m, 1.1, grad, scale);
    x = round((0.37 + 0.32*sin(pi*prog) + 0.1*prog)*Nx);
    y = round((0.22 + 0.66*prog)*Nx);
    rec1bg = zeros(Nx, Nx);
    rec1bg(x-n+1:x+n,y-m+1:y+m) = rec1bg(x-n+1:x+n,y-m+1:y+m) + 0.8*(rec .* hole);

    % "rectangle" (ball) 2
    ball = roundedRectangle(2*r, 2*r, 1.7, grad2);
    peg = roundedRectangle(2*r, 2*r, 2, 0.2, 0.4);
    x = round((0.37 + 0.42*prog)*Nx);
    y = round((0.80 - 0.53*prog)*Nx);
    rec2bg = zeros(Nx, Nx);
    rec2bg(x-r+1:x+r,y-r+1:y+r) = ball + peg;

    for i = 1:length(translations)
        ti = translations(i); % What translation to add
        switch abs(ti)
            case 1
                rec = rec1bg;
            case 2
                rec = rec2bg;
            otherwise
                error("Option '%d' not implemented!", t);
        end
        if ti < 0 % Mirror horizontally
            rec = fliplr(rec);
        end
        objCell{i + 2} = rec; % Add to list
    end
    
    % Deformation
    if deformation ~= 0
        prog2 = 2*t/T;
        e = sign(deformation);
        d = abs(deformation);
        st = (d * 0.05 * cos(pi * prog2) + 0.95)^e;
        invA = [1/st, 0;
                0,  st]; % Matrix
        b = 0.44 * [(1-st); 0]; % Translation

        nx = Nx/2;
        x = linspace(-nx+1, nx, Nx); % Centered coordinates, higher resolution
        y = x;
        % Apply inverse deformations
        Ax = invA(1,1)*(x - b(1)) + invA(1,2)*(y - b(2));
        Ay = invA(2,1)*(x - b(1)) + invA(2,2)*(y - b(2));

        % Bound to [1, Nx]
        indx = min(Nx, max(1, round(Ax + nx)));
        indy = min(Nx, max(1, round(Ay + nx)));

        objCell{1} = background(indx, indy);
        objCell{2} = rotLayer(indx, indy);
    end

    % Add background and all moving objects
    obj(:,:,t) = sum(cat(3,objCell{:}),3);
end
obj = min(1, obj ./ bgMax);

%% Helper functions
function rec = roundedRectangle(n, m, p, gradient, scale)
    % Function for creating a rounded n x m rectangle using the p-norm
    if nargin < 5
        scale = 1;
    end
    s = abs(1/scale);

    yy = linspace(-s,s,n);
    xx = linspace(-s,s,m);

    [XX, YY] = meshgrid(xx,yy);

    rec = (abs(XX).^p + abs(YY).^p).^(1/p) <= 1;

    if nargin >= 4
        rec = rec .* gradient;
    end
end

function P = fillDefaultParameters(params)
    % Function for setting default values to parameters not defined by
    % user in `params`

    % Default parameters
    P = [];
    % Background block
    P.blockSz = [0.11, 0.2];
    P.blockBrdVal = 0.6;
    P.blockVal = 0.35;
    P.blockBrd = 0.01;
    P.sigma = 5;

    % Top circles
    P.circRadii = [0.12, 0.14, 0.13]; % Radius
    P.circys = [0.22, 0.51, 0.78]; % y
    P.circxs = [0.63, 0.33, 0.45]; % x
    P.circps = [2, 1.4, 0.7]; % p-norm
    P.circVals = [0.45, 0.48, 0.6]; % Value

    P.circHoleSzs = [0.15, 0.25]; % Hole diameter
    P.circHoleDist = 0.2; % Distance between holes

    P.circGradVal = 1.2; % Gradient multiplier

    P.waveParams = [0.25, 5, 3]; % Strength, freq1, freq2

    % Movers
    P.m1sz = [0.09, 0.1];
    P.m2sz = 0.05;
    P.movementLimit = 0.34;

    % Set user given parameter values
    NdefaultParams = length(P);
    par = fieldnames(params);
    for iii = 1:length(par)
        P.(par{iii}) = params.(par{iii});
    end
    if length(P) > NdefaultParams
        warning("There should be %d parameters but now there are %d. Likely some field names are wrong!", NdefaultParams, length(P));
    end
end
end
%%%%%% simulate_dynamic_data.m %%%%%%
%
% Code for constructing dynamic CT data
% 
%%%%%%
% 
% Requirements:
% ASTRA Toolbox
% https://www.astra-toolbox.com/
% Recommended v1.9 or higher
%
% Spot - A Linear-Operator Toolbox
% https://www.cs.ubc.ca/labs/scl/spot/
% Recommended v1.2
% 
%%%%%%
%
% Created 23.9.2024 - Last edited 23.9.2024
% Tommi Heikkilä
% LUT University

% Clear workspace
clearvars
close all

%% Set data dimensions and other parameters

saveStuff = 0;
name = "all"; % Data name

Nx = 256;

angles = 0:1:359;
T = 180; % Number of time steps

Ndet = 280;

eps = 0.03; % Noise level

thetas = reshape(angles, [], T);
objBig = createDynamicObj(2*Nx, T, 'deformation', 1.5, 'rotation', 15, 'translations', 1);

figure; montage(objBig);


%% Animation 
tik = 3/T;

for t = 1:T
    figure(101);
    imagesc(objBig(:,:,t));
    axis equal
    axis off
    colorbar
    pause(tik)
    drawnow
end

%% Create forward operator

vec = @(x) x(:);

obj = zeros(Nx, Nx, T);

sinogram = zeros(length(angles), Ndet);

for t = 1:T
    theta = deg2rad(thetas(:,t));
    n = length(theta);
    A = create_ct_operator_2d_parallel_astra_cpu(2*Nx, 2*Nx, 2*Ndet, theta);
    Mvec = A*vec(objBig(:,:,t));
    M = reshape(Mvec, n, 2*Ndet); % Upsampled short sinogram
    
    sinogram((t-1)*n+1:t*n, :) = 0.25*(M(:, 1:2:end-1) + M(:, 2:2:end));

    obj(:,:,t) = imresize(objBig(:,:,t), [Nx, Nx]);
end

sinogram = sinogram + max(sinogram(:))*eps*randn(size(sinogram));

figure;
imagesc(sinogram);
title("Final sinogram")

%% Let's try it

anglesRad = deg2rad(angles);
A = create_ct_operator_2d_parallel_astra_cpu(Nx, Nx, Ndet, anglesRad);

bp = reshape(A'*sinogram(:), Nx, Nx);

figure;
imagesc(bp);
colorbar

%% Save

if saveStuff
    fprintf("Saving data! \n");
    clearvars -except anglesRad sinogram obj eps name
    save(sprintf("data/simData_%s_256x256x180_parallel.mat", name));
end

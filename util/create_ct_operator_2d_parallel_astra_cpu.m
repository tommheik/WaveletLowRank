function A = create_ct_operator_2d_parallel_astra_cpu( xDim, yDim, nDet, angles )
%CREATE_CT_OPERATOR_2D_PARALLEL_ASTRA_CPU Create 2D CT forward model
%   A = create_ct_operator_2d_parallel_astra( xDim, yDim, nDet, angles ) computes the 
%   forward model, i.e. X-ray projection operator, for the 2D parallel-beam CT
%   project given in input parameters. The x- and y-dimensions of the CT volume 
%   are given by parameters ''xDim'' and ''yDim'', respectively. The imaging 
%   geometry is created using the number of detector elements ''nDet'' and
%   given projection angles ''angles'' (in radians!)
%   It is assumed that a flat detector has been used for the X-ray projection 
%   measurements. The pixel size (in mm) is fixed for simplicity.
%
%   The forward model is an operator that behaves like a matrix, for
%   example in operations like A*x and and A.'*x, but no explicit matrix is
%   actually created.
%
%   Use of this function requires that the ASTRA Tomography Toolbox 
%   (https://www.astra-toolbox.com/) and the Spot Linear-Operator Toolbox 
%   (https://www.cs.ubc.ca/labs/scl/spot/) have been added to the MATLAB 
%   path.
%
%   This function is adapted from the HelTomo Toolbox, which was created 
%   primarily for use with CT data measured in the Industrial Mathematics 
%   Computed Tomography Laboratory at the University of Helsinki.
%
%   T. Heikkilä
%   Created:            9.12.2022
%   Last edited:        5.9.2024
%   
%   Based on codes by 
%   Alexander Meaney, University of Helsinki

% Validate input parameters

if ~isscalar(xDim) || xDim < 1 || floor(xDim) ~= xDim
    error('Parameter ''xDim'' must be a positive integer.');
end

if ~isscalar(yDim) || yDim < 1 || floor(yDim) ~= yDim
    error('Parameter ''yDim'' must be a positive integer.');
end

% ASTRA code begins here
fprintf('Creating geometries and data objects in ASTRA... ');

% Create volume geometry, i.e. reconstruction geometry
volumeGeometry = astra_create_vol_geom(xDim, yDim);

pixelSize = 1.0; % Magic number

% Create projection geometry
projectionGeometry = astra_create_proj_geom('parallel', pixelSize, nDet, angles);
% Create the Spot operator for ASTRA using the GPU.
A = opTomo('strip', projectionGeometry, volumeGeometry);

fprintf('done.\n');

% Memory cleanup
astra_mex_data2d('delete', volumeGeometry);
astra_mex_data2d('delete', projectionGeometry);
clearvars -except A

end
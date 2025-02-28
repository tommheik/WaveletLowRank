function [f, iter, info] = PDFPalgorithm(A, m, T, param)
    % Main PDFP algorithm for reaching the solution iteratively
    fprintf('----------\nBegin PDFP + cylSh algorithm with parameter tuning!\n----------\n')
    tic;
    %% Unload PDFP parameters
    if isfield(param,'maxIter'); maxIter = param.maxIter; else; maxIter = 500; fprintf('Using default value for maxIter\n'); end
    if isfield(param,'normTol'); normTol = param.normTol; else; normTol = 1e-5; fprintf('Using default value for normTol\n'); end
    if isfield(param,'sparTol'); sparTol = param.sparTol; else; sparTol = 1e-2; fprintf('Using default value for sparTol\n'); end
    if isfield(param,'plotFreq'); plotFreq = param.plotFreq; else; plotFreq = 10; fprintf('Using default value for plotFreq\n'); end

    lambda = param.lambda;
    gamma = param.gamma;
    desiredSparsity = param.desiredSparsity;
    psi = param.psi;
    omega = param.omega;
    if isfield(param,'kappa'); kappa = param.kappa; else; kappa = 1e-7; fprintf('Using default value for kappa\n'); end

    % Initialize
    fLen = size(A, 2);
    if isfield(param,'f0'); f = param.f0; else; f = zeros(fLen,1); fprintf('Using default value for f0\n'); end
    Tf = T.fwd(f);
    fSz = param.xSz;

    % Number of coefficients
    coefLen = 0;
    for l = 1:length(Tf)
        coefLen = coefLen + numel(Tf{l});
    end

    relChange = nan(1,maxIter+1);
    dataFit = nan(1,maxIter);
    l1Norm = nan(1,maxIter);
    alphas = nan(1,maxIter);
    sparsity = nan(1,maxIter);
    e = 1;

    alpha = psi*1e-5;
    beta = omega*alpha; % Tuning step length
    iter = 1;
    alphas(1) = alpha;

    %% Iterate

    while (iter <= maxIter) && ((relChange(iter) > normTol) || (abs(e) > sparTol))   
        fOld = f;
        Af = A*f;
        dif = Af - m;
        BP = A'*dif;

        dataFit(iter) = norm(dif) / norm(m);

        % PDFP steps
        d = max(0, f - gamma*BP - lambda*T.adj(Tf));
        Td = T.fwd(d);
        Tf = cellfun(@plus, Tf, Td, 'UniformOutput', false);
        Tf = IdMinusSoftThreshold(Tf, alpha*gamma/lambda);
        f = max(0, f - gamma*BP - lambda*T.adj(Tf));

        Tf = T.fwd(f); % Transform the current iterate
        spar = currentSparsity(Tf, coefLen, kappa); % Compute current sparsity

        sparsity(iter) = spar;
        l1Norm(iter) = sum(cellfun(@(x) norm(x(:),1), Tf));
        relChange(iter+1) = norm(f - fOld) / norm(fOld);

        eOld = e; % Old difference in sparsity
        e = spar - desiredSparsity; % Update

        % Change beta if controller error e changes sign
        if sign(e) ~= sign(eOld)
            beta = beta*(1-abs(e-eOld));
        end

        % Update alpha
        alpha = max(0, alpha + beta*e);
        alphas(iter) = alpha;

        if mod(iter,10) == 0
            fprintf('Iteration number %d reached \n', iter);
            fprintf('Relative change: %.5f, sparsity: %.3f \n', relChange(iter), spar);
            fprintf('----------\n')
        end
        if mod(iter,plotFreq) == 0
            figure(100)
            montage(reshape(f,fSz), 'DisplayRange', []);
            title(sprintf('Reconstruction at iter: %d', iter));

            figure(101)
            title(sprintf('Iteration: %d \n', iter));
            subplot(4,1,1)
            plot(sparsity(1:iter))
            ylabel('Sparsity')
            yline(desiredSparsity, 'r');

            subplot(4,1,2)
            semilogy(alphas(1:iter))
            ylabel('alpha')

            subplot(4,1,3)
            semilogy(relChange(1:iter))
            ylabel({'Relative'; 'change'})
            yline(normTol, 'r');

            subplot(4,1,4)
            semilogy(0.5*dataFit(1:iter).^2 + alphas(1:iter).*l1Norm(1:iter))
            ylabel({'Functional'; 'value'})
            xlabel('Iteration')
        end

        iter = iter+1;
    end
    iter = iter - 1;
    timeTot = toc;
    
    fprintf('Total computational time: %.1f s, approximately %.2f s per iteration \n', timeTot, timeTot / iter);
    
    if iter == maxIter
        fprintf('Maximum iteration count reached! Iteration stopped \n');
        relChange = relChange(2:end);
    else
        fprintf('Stopping criterion reached after %d iterations! \n', iter);
        % Cut stored arrays to correct length
        relChange = relChange(2:iter+1);
        dataFit = dataFit(1:iter);
        l1Norm = l1Norm(1:iter);
        alphas = alphas(1:iter);
        sparsity = sparsity(1:iter);
    end
    info.relChange = relChange;
    info.dataFit = dataFit;
    info.l1Norm = l1Norm;
    info.alphas = alphas;
    info.sparsity = sparsity;
    info.functionalValues = 0.5*dataFit.^2 + alphas.*l1Norm;
    info.timeTot = timeTot;
end

function C = IdMinusSoftThreshold(C, alpha)
    %%% Perform I - S_a(C), where I is the identity, S_a is the soft
    %%% thresholding operator and C is a coefficient structure
    
    % Coefficients are stored in a cell array
    % C can be complex valued
    for iii = 1:length(C)
        c = C{iii};
        % If |c| > alpha, then c - S_a(c) = arg(c)*alpha, 
        % otherwise c - S_a(c) = c - 0 = c;
        if all(isreal(c))
            c(c > alpha) = alpha;
            c(c < -alpha) = -alpha;
        else
            a = angle(c(abs(c) > alpha));
            c(abs(c) > alpha) = alpha*exp(1j*a);
        end
        C{iii} = c;
    end
end

function cs = currentSparsity(c, coefLen, kappa)
    %%% Compute ratio of "nonzero" coefficients
    
    % Count the number of coefficients c such that |c| >= kappa in each
    % cell and sum them together
    Nbig = 0;
    for iii = 1:length(c)
        Nbig = Nbig + nnz(abs(c{iii}) >= kappa);
    end
    
     % Current sparsity is a ration of "big" coefficients wrt. total number of coefficients
    cs = Nbig / coefLen;
end
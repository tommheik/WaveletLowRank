function y = SoftThresh(x,mu)
    %%% Soft thresholding function
    if ~isreal(x)
        warning('We expect to only threshold real variables!')
    end
    y = (x - mu).*(x > mu) + (x + mu).*(x < -mu);
end 
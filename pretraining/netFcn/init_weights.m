function weights = init_weights(sz, bias)
    if nargin < 2 || bias
        weights{1} = 0.01 * randn(sz, 'single');
        weights{2} = single(0.1) * ones(sz(4), 1);
    else
        weights{1} = 0.01 * randn(sz);
    end
end
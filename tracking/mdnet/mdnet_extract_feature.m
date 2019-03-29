function featrs = mdnet_extract_feature(net, ims, opts)
    net.mode = 'test';
    n = size(ims, 4);
    
    numBatches = ceil(n / opts.batchSize4Test);
    
    for i = 1:numBatches
        batchStart = opts.batchSize4Test * (i-1) + 1;
        batchEnd = opts.batchSize4Test * i;
        batch = ims(:, :, :, batchStart:min(end, batchEnd));
        if opts.useGpu
            batch = gpuArray(batch);
        end
        
        net.eval({'input', batch});
        
        f = net.vars(end).value ;
        if ~exist('featrs', 'var')
            if strcmp(net.device, 'gpu')
                featrs = gpuArray.zeros(size(f,1), size(f,2), size(f,3), n, 'single');
            else
                featrs = zeros(size(f,1), size(f,2), size(f,3), n, 'single');
            end
        end
        featrs(:, :, :, batchStart:min(end, batchEnd)) = f;
    end
end
function state = mdnet_state_initialize(img, region, opts)
    % opts
    [net_c, net_f] = mdnet_net_initialize(opts);
    state.net_c = net_c;
    state.net_f = net_f;
    state.net_c.mode = 'test'; 
    state.net_f.mode = 'test';
    
    if opts.useGpu
        state.net_c.move('gpu');
        state.net_f.move('gpu');
        opts.averageImage = gpuArray(opts.averageImage);
    end

    opts.imageSize = size(img); 
    opts.gridGenerator = ...
        dagnn.AffineGridGenerator('Ho', opts.inputSize(1), ...
                                  'Wo', opts.inputSize(2)); 
    
    state.opts = opts;
    
    state.netOutIdx = state.net_f.getVarIndex('prediction');
    state.result = region;    
    state.targetRect = region;
    
    state.targetScore = 1;
    state.estimated = [];
    state.initConf = 0;
    
    state.currFrame = 1;
    state.succIndex = 1;
    state.trans_f = state.opts.trans_f;
    state.scale_f = state.opts.scale_f;
end
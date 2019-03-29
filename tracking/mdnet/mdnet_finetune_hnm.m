function net = mdnet_finetune_hnm(net, posData, negData, varargin)

    opts.verbose = false;
    opts.useGpu = true;
    opts.conserveMemory = true ;
    opts.sync = true ;

    opts.maxIters = 30;
    opts.learningRate = 0.001;
    opts.weightDecay = 0.0005 ;
    opts.momentum = 0.9 ;
    opts.initialize = false;

    opts.batchSize4HNM = 1024;
    opts.backPropDepth = inf;
    
    opts.batchSize = 128;
    opts.batchPos = 32;
    opts.batchNeg = 96;
    
    opts.solver = [];
    opts.solverOpts = opts.solver();
    opts.nesterovUpdate = false;
    [opts, varargin] = vl_argparse(opts, varargin) ;
    
    % ---------------------------------------------------------------------
    %                                                Network initialization
    % ---------------------------------------------------------------------
    state.solverState = cell(1, numel(net.params)) ;
    state.solverState(:) = {0} ;

    % -----------
    % Initilizing
    if opts.useGpu
        one = gpuArray(single(1)) ;
    else
        one = single(1) ;
    end

    numPos = size(posData, 4);
    numNeg = size(negData, 4);

    posData = posData(:,:,:,prepare_data_list(0, numPos, opts.batchPos, opts.maxIters));
    negData = negData(:,:,:,prepare_data_list(0, numNeg, opts.batchSize4HNM, opts.maxIters));
    
    % objective fuction
    obj = zeros(1, opts.maxIters);
    pPred = net.getVarIndex('prediction');
    pLoss = net.getVarIndex('loss');
    pLabl = net.getVarIndex('label');
    
    % training on training set
    net.reset();
    for t = 1:opts.maxIters
        if opts.verbose
            fprintf('\ttraining batch %3d of %3d ... ', t, opts.maxIters) ;
        end
        excuTime = tic ;
        % --------------------
        % hard negative mining
        % --------------------
        bstart = (t - 1) * opts.batchSize4HNM + 1;
        bend = t * opts.batchSize4HNM;
        batchNegData = negData(:, :, :, bstart:bend);
        net.mode = 'test';
        if opts.useGpu
            batchNegData = gpuArray(batchNegData);
        end
        net.eval({'input', batchNegData});
        prediction = squeeze(gather(net.vars(pPred).value(:,:,2,:)));
        [prediction, order] = sort(prediction, 'descend');
        batchNegData = batchNegData(:, :, :, order(1:opts.batchNeg));
        bstart = (t-1) * opts.batchPos + 1;
        bend = t * opts.batchPos;
        batchPosData = posData(:, :, :, bstart:bend);
        
        batchData = cat(4, batchPosData, batchNegData);
        if opts.useGpu
            batchData = gpuArray(batchData);
        end
        batchLabel = [2*ones(opts.batchPos, 1, 'single'); ones(opts.batchNeg, 1, 'single')];
        batchInputs = cat(2, {'input', batchData}, {'label', batchLabel});
 
        net.mode = 'normal';
        net.eval(batchInputs, {'loss', one});
        net.vars(pLabl).value = [];
        state = accumulate_gradients(net, state, opts, numel(batchLabel), []);
        obj(t) = gather(net.vars(pLoss).value) / numel(batchLabel);
        excuTime = toc(excuTime);

        if opts.verbose
            fprintf('network training batch %3d of %3d ---- obj %.3f, %.3fs\n', ...
                t, opts.maxIters, obj(t), excuTime) ;
        end    
    end 
    
    net.mode = 'test';
end


function dataPerm = prepare_data_list(perms, nums, batch, maxIters)
    dataPerm = [];
    remain = batch * maxIters;
    while(remain > 0)
        if(perms == 0)
            dataList = randperm(nums)';
        end
        dataPerm = cat(1, dataPerm, dataList(perms+1 : min(end, perms+remain)));
        perms = min(length(dataList), perms + remain);
        perms = mod(perms, length(dataList));
        remain = batch * maxIters - length(dataPerm);
    end
end
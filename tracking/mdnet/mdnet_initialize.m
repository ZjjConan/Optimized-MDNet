function state = mdnet_initialize(state, img)

    if state.opts.useGpu
        img = gpuArray(single(img));
    else
        img = single(img);
    end
    
    img = sub_average_img(img, state.opts);
    
    if(state.opts.bbreg)
        posSamples = generate_samples('uniform_aspect', state.targetRect, state.opts.bbregNums * 10, state.opts, 0.3, 10);
        r = bboxOverlapRatio(posSamples, state.targetRect);
        posSamples = posSamples(r>0.6, :);
        posSamples = posSamples(randsample(end, min(state.opts.bbregNums, end)), :);
        ims = mdnet_bilinear_crop(img, posSamples, state.opts);        
        featrs = mdnet_extract_feature(state.net_c, ims, state.opts);
        
        X = permute(gather(featrs), [4,3,1,2]);
        X = X(:,:);
        bbox = posSamples;
        gts = repmat(state.targetRect, size(posSamples, 1), 1);
        state.regressor = train_bbox_regressor(X, bbox, gts);
    end

    % ----------------------------
    % Extracting training examples
    % ----------------------------
    % draw positive/negative samples
    posExamples = generate_samples('gaussian', state.targetRect, state.opts.initNumPos * 2, state.opts, 0.1, 5);
    r = bboxOverlapRatio(posExamples, state.targetRect);
    posExamples = posExamples(r > state.opts.initThrPos, :);
    posExamples = posExamples(randsample(end, min(state.opts.initNumPos, end)), :);

    negExamples = [generate_samples('uniform', state.targetRect, state.opts.initNumNeg, state.opts, 1, 10);...
                   generate_samples('whole', state.targetRect, state.opts.initNumNeg, state.opts)];
    r = bboxOverlapRatio(negExamples, state.targetRect);
    negExamples = negExamples(r < state.opts.initThrNeg, :);
    negExamples = negExamples(randsample(end, min(state.opts.initNumNeg, end)), :);

    examples = [posExamples; negExamples];
    posIdx = 1:size(posExamples,1);
    negIdx = (1:size(negExamples,1)) + size(posExamples,1);

    % extract conv3 features
    ims = mdnet_bilinear_crop(img, examples, state.opts);
    featrs = mdnet_extract_feature(state.net_c, ims, state.opts);
    posData = featrs(:, :, :, posIdx);
    negData = featrs(:, :, :, negIdx);


    % ------------
    % Learning CNN
    % ------------
    state.net_f = ...
        mdnet_finetune_hnm(state.net_f, posData, negData, state.opts, ...
                           'maxIters', state.opts.initMaxIters, ...
                           'learningRate', state.opts.initLr, ...
                           'batchSize', state.opts.initBatchSize, ...
                           'batchPos', state.opts.initBatchPos, ...
                           'batchNeg', state.opts.initBatchNeg, ...
                           'batchSize4HNM', state.opts.initBatchSize*8);
    % ---------------------------------------
    % Prepare training data for online update
    % ---------------------------------------
    state.totalPosData = {};
    state.totalNegData = {};
    
    negExamples = generate_samples('uniform', state.targetRect, state.opts.updateNumNeg * 2, state.opts, 2, 5);
    r = bboxOverlapRatio(negExamples, state.targetRect);
    negExamples = negExamples(r<state.opts.initThrNeg, :);
    negExamples = negExamples(randsample(end, min(state.opts.updateNumNeg, end)), :);
    
    ims = mdnet_bilinear_crop(img, negExamples, state.opts);
    featrs = mdnet_extract_feature(state.net_c, ims, state.opts);
    state.totalPosData{1} = posData;
    state.totalNegData{1} = featrs;
end


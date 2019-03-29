function state = mdnet_track(state, img)

    state.currFrame = state.currFrame + 1;
    
    if state.opts.useGpu
        img = gpuArray(single(img));
    else
        img = single(img);
    end
    
    img = sub_average_img(img, state.opts);
    
    % ----------
    % Estimation
    % ----------
    % draw target candidates
    samples = generate_samples('gaussian', state.targetRect, ...
                               state.opts.numSamples, state.opts, ...
                               state.trans_f, state.scale_f);

    ims = mdnet_bilinear_crop(img, samples, state.opts);
    featrs = mdnet_extract_feature(state.net_c, ims, state.opts);
    
    % evaluate the candidates
    if state.opts.useGpu, featrs = gpuArray(featrs); end
    state.net_f.eval({'input', featrs});
    predictions = gather(squeeze(state.net_f.vars(state.netOutIdx).value(:,:,2,:)));
    [predictions, order] = sort(predictions, 'descend');
    targetScore = mean(predictions(1:5));
    targetRect = round(mean(samples(order(1:5), :)));
    
    % final target
    state.result(state.currFrame, :) = targetRect;
    state.targetRect = targetRect;
    state.targetScore = targetScore;
    
    % extend search space in case of failure
    if (targetScore < 0)
        state.trans_f = min(1.5, 1.1*state.trans_f);
    else
        state.trans_f = state.opts.trans_f;
    end

    % sometimes has negative impact with bbreg
    % bbox regression
    if (state.opts.bbreg && targetScore > 0)
        X = permute(gather(featrs(:, :, :, order(1:5))), [4,3,1,2]);
        X = X(:,:);
        bbox = samples(order(1:5),:);
        bbox = predict_bbox_regressor(state.regressor.model, X, bbox);
        state.result(state.currFrame, :) = round(mean(bbox, 1));
    end
    
    % ---------------------
    % Prepare training data
    % ---------------------
    if (targetScore > 0)
        posExamples = generate_samples('gaussian', state.targetRect, state.opts.updateNumPos * 2, state.opts, 0.1, 5);
        r = bboxOverlapRatio(posExamples,  state.targetRect);
        posExamples = posExamples(r > state.opts.updateThrPos, :);
        posExamples = posExamples(randsample(end, min(state.opts.updateNumPos, end)), :);

        negExamples = generate_samples('uniform',  state.targetRect, state.opts.updateNumNeg * 2, state.opts, 2, 5);
        r = bboxOverlapRatio(negExamples,  state.targetRect);
        negExamples = negExamples(r < state.opts.updateThrNeg, :);
        negExamples = negExamples(randsample(end, min(state.opts.updateNumNeg, end)), :);

        examples = [posExamples; negExamples];
        posIdx = 1:size(posExamples,1);
        negIdx = (1:size(negExamples,1)) + size(posExamples,1);

        ims = mdnet_bilinear_crop(img, examples, state.opts);
        featrs = mdnet_extract_feature(state.net_c, ims, state.opts);

        state.totalPosData{state.currFrame} = featrs(:, :, :, posIdx);
        state.totalNegData{state.currFrame} = featrs(:, :, :, negIdx);

        state.succIndex = [state.succIndex, state.currFrame];
        if (numel(state.succIndex) > state.opts.numFramesLT)
            state.totalPosData{state.succIndex(end - state.opts.numFramesLT)} = single([]);
        end
        if (numel(state.succIndex) > state.opts.numFramesST)
            state.totalNegData{state.succIndex(end - state.opts.numFramesST)} = single([]);
        end
    else
        state.totalPosData{state.currFrame} = single([]);
        state.totalNegData{state.currFrame} = single([]);
    end
end
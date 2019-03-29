function inputs = get_batch_imgs(opts, imdb, batch)
    
    video = imdb.images.data{batch};
    frame = randperm(numel(video), opts.numFrames);
    
    if opts.useGpu
        ims = vl_imreadjpeg(video(frame), 'NumThreads', 4, ...
            'Pack', 'Resize', imdb.images.imsz(batch, :), 'GPU');
    else
        ims = vl_imreadjpeg(video(frame), 'NumThreads', 4, ...
            'Pack', 'Resize', imdb.images.imsz(batch, :));
    end
    
    ims = prepare_imgs(ims{1}, opts);
    
    pbox = vertcat(imdb.images.bbox{batch}.pos{frame});
    nbox = vertcat(imdb.images.bbox{batch}.neg{frame});
    
    psel = randperm(size(pbox, 1), opts.batchPos);
    nsel = randperm(size(nbox, 1), opts.batchNeg);
    
    bbox = single([pbox(psel, :); nbox(nsel, :)]);

    if opts.useGpu
        bbox = gpuArray(single(bbox));
        imo = gpuArray.zeros(opts.inputSize(1), opts.inputSize(2), 3, ...
            size(bbox, 1), 'single');
    else
        ims = single(ims{1});
        bbox = single(bbox);
        imo = zeros(opts.inputSize(1), opts.inputSize(2), 3, ...
            size(bbox, 1), 'single');
    end
    
    opts.imageSize = size(ims);
    
    pos = bbox(:, 2:3) + bbox(:, 4:5)/2;
    sca = opts.inputSize / (opts.inputSize - opts.padding*2);
    sca = bbox(:, 4:5) * sca;
    
    grids = generate_bilinear_grids(pos', sca', opts);
    

    for i = 1:size(ims, 4)
        idx = bbox(:, 1) == frame(i);
        imo(:, :, :, idx) = vl_nnbilinearsampler(ims(:,:,:,i), grids(:,:,:,idx));
    end

    label = single([2*ones(1, opts.batchPos), ones(1, opts.batchNeg)]);
    
    if rand > 0.5
        inputs = {'input', fliplr(imo), 'label', label, 'k', batch};
    else
        inputs = {'input', imo, 'label', label, 'k', batch};
    end
end


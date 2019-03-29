function pimgs = mdnet_bilinear_crop(img, boxes, opts)
    if opts.useGpu
        img = gpuArray(img);
        boxes = gpuArray(boxes);
    end
    
    bpos = boxes(:, 1:2) + boxes(:, 3:4)/2;
    gsiz = boxes(:, 3:4) * opts.inputSize(1) / (opts.inputSize(1) - opts.padding*2);

    grids = generate_bilinear_grids(bpos', gsiz', opts);
    pimgs = vl_nnbilinearsampler(img, grids);
end
function ims = prepare_imgs(ims, varargin)

    opts.augGray = false;
    opts.augGrayProb = 0.1;
    opts.averageImage = single(128);
    [opts, varargin] = vl_argparse(opts, varargin);
    
    if opts.augGray
        if rand < opts.augGrayProb
            imo = cell(size(ims, 4), 1);
            for i = 1:size(ims, 4)
                imo{i} = rgb2gray(uint8(ims(:,:,:,i)));
            end
            imo = cat(4, imo{:});
            ims = single(repmat(imo, [1 1 3 1]));
        end
    end

    % mean average image
    ims = sub_average_img(ims, opts);
end


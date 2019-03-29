function imdb_new = mdnet_sample_bbox(imdb, varargin)
    
    opts.useGpu = false;
    opts.posPerFrame = 50;
    opts.negPerFrame = 200;
    opts.scaleFactor = 1.05;
    opts.posRange = [0.7 1];
    opts.negRange = [0 0.5];
    opts.debug = false;
    
    [opts, varargin] = vl_argparse(opts, varargin);
    
    imdb_new = imdb;
    
    if isempty(imdb.images.bbox{1}.gts)
        error('no ground truth in current imdb');
    end
    
    imdb_new.meta.posPerFrame = opts.posPerFrame;
    imdb_new.meta.negPerFrame = opts.negPerFrame;
    imdb_new.meta.posRange = [0.7 1];
    imdb_new.meta.negRange = [0 0.5];
    imdb_new.meta.scaleFactor = opts.scaleFactor;
    
    tic
    for v = 1:numel(imdb_new.images.data)
        config.imgList = imdb_new.images.data{v};
        config.gt = imdb_new.images.bbox{v}.gts;
        
        roidb = seq_to_roidb(config, opts);
        imdb_new.images.bbox{v}.pos = {roidb(:).pos_boxes}';
        imdb_new.images.bbox{v}.neg = {roidb(:).neg_boxes}';
        fprintf('%s: sampling %d / %d videos time %.2fs\n', ...
            mfilename, v, numel(imdb_new.images.data), toc);
        
        if opts.debug
            for i = 1:min(5, numel(config.imgList))
                imshow(vaReadImage(config.imgList{i}, true));
                % draw postive bounding box
                for b = 1:5
                    rectangle('Position', imdb_new.images.bbox{v}.pos{i}(b, 2:end), ...
                        'LineWidth', 1.5, 'EdgeColor', 'g');
                end
                
                % draw negtive bounding box
                for b = 1:20
                    rectangle('Position', imdb_new.images.bbox{v}.neg{i}(b, 2:end), ...
                        'LineWidth', 1.5, 'EdgeColor', 'r');
                end
            end
        end
    end
end


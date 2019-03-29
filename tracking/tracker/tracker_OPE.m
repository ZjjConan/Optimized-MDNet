function [result, fps] = tracker_OPE(cfg, opts)

    tprocess.time.all = 0;
    tprocess.time.det = 0;
    tprocess.time.dup = 0;
    
    img = read_img(cfg.img_files{1});
    
    state = opts.state_initialize(img, cfg.ground_truth(1, :), opts);
    
    % times
    for f = 1:numel(cfg.img_files)
        img = read_img(cfg.img_files{f});
        
        time_all = tic;
        
        if f == 1
            state = opts.initialize(state, img);
        else
            time_det = tic;
            state = opts.track(state, img);
            tprocess.time.det = tprocess.time.det + toc(time_det);
            
            time_dup = tic;
            state = opts.update(state);
            tprocess.time.dup = tprocess.time.dup + toc(time_dup);
        end
        time_tmp = toc(time_all);
        tprocess.time.all = tprocess.time.all + time_tmp;
        
        if state.opts.verbose
            if state.currFrame == 1
                 videoPlayer = vision.VideoPlayer('Position', [100 100 [size(img, 2), size(img, 1)]+30]);
            end
            
            img = gather(single(img))/255;
            img = insertShape(img, 'Rectangle', state.result(f, :), 'LineWidth', 4, 'Color', 'green');
            img = insertText(img, [10 10], ['FPS: ' num2str(1/time_tmp)], 'TextColor', 'y', ...
                'FontSize', 20, 'BoxColor', 'black', 'BoxOpacity', 0.8);
            % Display the annotated video frame using the video player object.
            step(videoPlayer, img);
        end
    end
    
    if state.opts.useGpu
        state.net_c.move('cpu');
        state.net_f.move('cpu');
    end
    
    result = state.result;
        
    fps.all = numel(cfg.img_files) / tprocess.time.all;
    fps.det = numel(cfg.img_files) / tprocess.time.det;
    fps.dup = numel(cfg.img_files) / tprocess.time.dup;
end
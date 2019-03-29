function opts = mdnet_get_opts(varargin)

    % set opts
    opts.verbose = true;   
    % use gpu
    opts.useGpu = true;
    % model def
    opts.netFile = './model/';
    % test policy
    opts.batchSize4Test = 128;
    
    % bounding box regression
    opts.bbreg = true;
    opts.bbregNums = 1000;
    
    % learning policy
    opts.initBatchSize = 128; %128;
    opts.initBatchPos = 32; %32;
    opts.initBatchNeg = 96; %96;
    
    % initial training policy
    opts.initLr = 0.001; % x10 for fc6
    opts.initMaxIters = 30;
    
    opts.initNumPos = 500;
    opts.initNumNeg = 5000;
    opts.initThrPos = 0.7;
    opts.initThrNeg = 0.5;
    
    % learning policy
    opts.updateBatchSize = 128;
    opts.updateBatchPos = 32;
    opts.updateBatchNeg = 96;
    
    % update policy
    opts.updateLr = 0.003;
    opts.updateMaxIters = 10;
    
    opts.updateNumPos = 50; %50;
    opts.updateNumNeg = 200; %200;
    opts.updateThrPos = 0.7;
    opts.updateThrNeg = 0.3;
    
    opts.updateInterval = 10; %10; % interval for long-term update
    opts.updateThreshold = 0; 
    
    % data gathering policy
    opts.numFramesLT = 100;  % long-term period
    opts.numFramesST = 20;  % short-term period
    
    % cropping policy
    opts.inputSize = [107 107];
    opts.padding = 16;
    
    % scaling policy
    opts.scaleFactor = 1.05;

    % sampling policy
    opts.numSamples = 256;
    opts.trans_f = 0.6;  % translation std: mean(width,height)*trans_f/2
    opts.scale_f = 1;  % scaling std: scale_factor^(scale_f/2)

    % average image
    opts.averageImage = single(reshape([122.6769, 116.67, 104.01], 1, 1, 3));
    
    [opts, varargin] = vl_argparse(opts, varargin);
end


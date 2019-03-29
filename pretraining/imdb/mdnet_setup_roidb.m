function [imdb, opts] = mdnet_setup_roidb(varargin)
    % -----------------------
    % training config setting
    % -----------------------
    opts.benchmarkSeqHome = '';

    % The list of tracking sequences for training MDNet.
    opts.seqsList  = {
        struct('dataset', 'vot2013', 'list', 'pretraining/seqList/vot13-otb.txt'),...
        struct('dataset', 'vot2014', 'list', 'pretraining/seqList/vot14-otb.txt'),...
        struct('dataset', 'vot2015', 'list', 'pretraining/seqList/vot15-otb.txt')
    };

    % The directory to store the RoIs for training MDNet.
    opts.imdbPath = '';
    
    % use temporal information
    opts.useTemporal = false;
    
    opts.cropMode         = 'warp';
    opts.numFetchThreads  = 8 ;
    opts.posRange         = [0.7   1];
    opts.negRange         = [0   0.5];
    
    opts.posPerFrame       = 50;
    opts.negPerFrame       = 200;
    opts.scaleFactor       = 1.05;
    opts.flip              = false;
    
    opts.dataSetup         = true;
    
    [opts, varargin] = vl_argparse(opts, varargin);
%     opts.roiPath = fullfile(opts.roiDir, 'roidb.mat');
    
    if ~opts.useTemporal
        opts.seq2roidb = @seq2roidb;
    else
        opts.seq2roidb = @seq2roidb_temporal;
    end
    
    % ----------------------
    % Sampling training data
    % ----------------------
%     genDir(opts.roiDir);
    
    if exist(opts.imdbPath, 'file') && ~opts.dataSetup
        imdb = load(opts.imdbPath) ;
    else
        imdb = mdnet_setup_data(opts.seqsList, opts);
        
%         imdb.images.data = cell(numel(tmpl), 1);
%         imdb.images.bbox = cell(numel(tmpl), 1);
%         imdb.images.set = ones(numel(tmpl), 1);
%         imdb.images.imsz = zeros(numel(tmpl), 2);
%         for i = 1:numel(tmpl)
%             imdb.images.data{i} = {tmpl{i}.img_path}';
%             imdb.images.bbox{i}.pos = {tmpl{i}.pos_boxes}';
%             imdb.images.bbox{i}.neg = {tmpl{i}.neg_boxes}';
%             info = imfinfo(tmpl{i}(1).img_path);
%             imdb.images.imsz(i, :) = [info.Height, info.Width];
%         end
        
        imdb.meta.normalization.averageImage = 128;
        
        save(opts.imdbPath, '-struct', 'imdb') ;
    end
end


% -------------------------------------------------------------------------
function roidb = mdnet_setup_data(seqsList, opts)
% -------------------------------------------------------------------------
    roidb = {};
    numFrames = 0;
    for D = 1:length(seqsList)

        dataset = seqsList{D}.dataset;
        seqs_train = importdata(seqsList{D}.list);
        
        roidb_ = cell(1,length(seqs_train));

        for i = 1:length(seqs_train)
            seq = seqs_train{i};
            fprintf('sampling %s:%s ... \n', dataset, seq);

            config = genConfig(dataset, seq, opts.benchmarkSeqHome);
            roidb_{i} = opts.seq2roidb(config, opts);
            % Display samples
%             figure(3);
%             for t = 1:length(config.imgList)
%                 imshow(config.imgList{t}); hold on;
%                 for j = 1:opts.negPerFrame
%                     rectangle('Position', roidb_{i}(t).neg_boxes(j,:), 'EdgeColor', [0 0 1], 'Linewidth', 1);
%                 end
%                 for j = 1:opts.posPerFrame
%                     rectangle('Position', roidb_{i}(t).pos_boxes(j,:), 'EdgeColor', [1 0 0], 'Linewidth', 1);
%                 end
%                 
%                 rectangle('Position', config.gt(t,:), 'EdgeColor', [0 1 0], 'Linewidth', 3);
%                 hold off;
%                 drawnow;
%                 pause(0.1);
%             end
            numFrames = numFrames + numel(config.imgList);
        end
        roidb = [roidb, roidb_];
    end
    fprintf('the total frames %d\n', numFrames);
end


% -------------------------------------------------------------------------
function genDir(path)
% -------------------------------------------------------------------------
    if ~exist(path,'dir')
        mkdir(path);
    end
end


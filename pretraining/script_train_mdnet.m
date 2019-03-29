% script for training mdnet
clear all; clc; close all
addpath('dagFcn');
addpath('netFcn');
addpath('trainFcn');

% Prepare a CNN model for learning MDNet (windows)
opts.initModel = 'D:/CNNModel/imagenet-vgg-m.mat';
opts.imdbPath = 'data/imdb_vot_otb.mat';
opts.expDir = 'data/snapshot/';
opts.outModelDir = '../models/mdnet_vot_otb.mat';

% load imdb
imdb = load(opts.imdbPath);

% network opts
opts.isDagNN = false;
opts.removePadding = true;
opts.multiDomainLearning = true;
opts.removeAfterThisLayer = 'conv4';
opts.trainNet = true;

net = prepare_model(opts, 'numBranches', numel(imdb.images.data));

% trainOpts
trainOpts.learningRate = 0.0001 * ones(1, 100) / 128;
trainOpts.numEpochs = numel(trainOpts.learningRate);
trainOpts.batchSize = 1;
trainOpts.epochSize = numel(imdb.images.data);
trainOpts.derOutputs = {'all_loss', 1};
trainOpts.gpus = [1];

batchOpts.numFrames = 8;
batchOpts.batchPos = 32;
batchOpts.batchNeg = 96;
batchOpts.useGpu = trainOpts.gpus >= 1;
batchOpts.averageImage = reshape([122.6769, 116.67, 104.01], 1, 1, 3);
batchOpts.inputSize = [107 107];
batchOpts.padding = 16;

batchOpts.gridGenerator = ...
    dagnn.AffineGridGenerator('Ho', batchOpts.inputSize(1), ...
                              'Wo', batchOpts.inputSize(2)); 

if opts.trainNet                  
    net = cnn_train_net(net, imdb, @(x, y) get_batch_imgs(batchOpts, x, y), ...
        'expDir', opts.expDir, trainOpts, 'val', find(imdb.images.set == 2), ...
        'continue', false, 'extractStatsFn', @extractStatsMDNet, ...
        'plotStatistics', false);
end
           
net = deploy_net(net);
net = net.saveobj();
save(opts.outModelDir, '-struct', 'net') ;
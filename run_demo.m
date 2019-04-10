clc; clear all; close all;

opts.netFile = 'models/mdnet_vot_otb.mat';
opts.trackerName = 'OptMDNet-Opt';
opts.videoPath = 'sequence/';
opts.videoAttr = 'demo';
opts.verbose = true;
opts.useGpu = 1;
opts.saveResult = false;
opts.videoList = [];
opts.settingFcn = @setting_mdnet_opt;
opts.trackerFcn = @tracker_OPE;

[~, opts.runFileName, ~] = fileparts(mfilename('fullpath'));
eval_tracker_OPE(opts);
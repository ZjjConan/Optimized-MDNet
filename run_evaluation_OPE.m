clc; clear all; close all;

opts.netFile = 'models/mdnet_vot_otb.mat';
opts.savePath = 'F:/Research/tracker_zoo/Evaluation/Results/OTB/';
opts.trackerName = 'OptMDNet-Opt';
opts.videoPath = 'D:/Dataset/Video/OTB/';
opts.videoAttr = 'OTB2015';
opts.verbose = false;
opts.useGpu = 1;
opts.saveResult = true;
opts.videoList = [];
opts.settingFcn = @setting_mdnet_opt;
opts.trackerFcn = @tracker_OPE;

[~, opts.runFileName, ~] = fileparts(mfilename('fullpath'));
eval_tracker_OPE(opts);
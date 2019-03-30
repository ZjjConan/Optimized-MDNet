% setup imdb for mdnet training

addpath('imdb');

% vot training for otb
% opts.saveDir = 'data/imdb_vot_otb.mat';
% opts.vidListDir = {'seqList', ...
%                    'seqList', ...
%                    'seqList'};
% opts.vidList = {'vot13-otb.txt', 'vot14-otb.txt', 'vot15-otb.txt'};
% opts.vidDir = {'D:/Dataset/Video/vot2013', 'D:/Dataset/Video/vot2014', 'D:/Dataset/Video/vot2015'};
% opts.vidName = 'vot';

% otb training for vot15
opts.saveDir = 'data/imdb_otb_vot.mat';
opts.vidListDir = {'seqList'};
opts.vidList = {'otb-vot15.txt'};
opts.vidDir = {'D:/Dataset/Video/OTB'};
opts.vidName = 'otb';

imdb = mdnet_setup_imdb(opts);

opts.useGpu = false;
opts.posPerFrame = 50;
opts.negPerFrame = 200;
opts.scaleFactor = 1.05;
opts.posRange = [0.7 1];
opts.negRange = [0 0.5];
opts.debug = false;

imdb = mdnet_sample_bbox(imdb);

save(opts.saveDir, '-struct', 'imdb', '-v7.3');
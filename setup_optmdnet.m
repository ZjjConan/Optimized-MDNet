function setup_optmdnet()
    clc; clear all; close all
    if ispc
        lib_path = 'D:/Libraries/';
    elseif isunix
        lib_path = '/media/zjjconan/Experiments/Libraries/'; 
    end

    matconvnet_path = fullfile(lib_path, 'matconvnet');
    run([matconvnet_path '/matlab/vl_setupnn']);
    
    root = fileparts(fileparts(mfilename('fullpath'))) ;
    addpath(fullfile(root, 'OptMDNet')) ;

    addpath(genpath(fullfile(root, 'OptMDNet/tracking')));
    addpath(fullfile(root, 'OptMDNet/utils'));
    addpath(fullfile(root, 'OptMDNet/vot'));
end
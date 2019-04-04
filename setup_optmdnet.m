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
    addpath(fullfile(root, 'Optimized-MDNet')) ;

    addpath(genpath(fullfile(root, 'Optimized-MDNet/tracking')));
    addpath(fullfile(root, 'Optimized-MDNet/utils'));
    addpath(fullfile(root, 'Optimized-MDNet/vot'));
end
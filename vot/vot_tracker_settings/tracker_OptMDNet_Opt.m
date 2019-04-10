% tracker config
tracker_label = 'OptMDNet_Opt';
tracker_interpreter = 'matlab';
tracker_trax = true;

tracker_path = 'F:/Research/tracker_zoo/Optimized-MDNet/';

netFile = 'F:/Research/tracker_zoo/Optimized-MDNet/models/mdnet_otb_vot.mat';
settingFcn = '@setting_mdnet_opt';
fullCommand = ['tracker_VOT(', '''' netFile, ''',', settingFcn, ')'];

% tracker command
tracker_command = generate_matlab_command(fullCommand, {tracker_path});

tracker_linkpath = {'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDNN/cuda-7.1/lib/x64/'};


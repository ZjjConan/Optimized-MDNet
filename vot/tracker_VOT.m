function tracker_VOT(netFile, settingFcn, varargin)

    cleanup = onCleanup(@() exit() );

    RandStream.setGlobalStream(RandStream('mt19937ar', 'Seed', sum(clock)));
    
try    
    setup_path_for_vot;
    
    opts.netFile = netFile;
    opts.verbose = 0;
    opts.useGpu = 1;
    
    runOpts = settingFcn(opts);
    [runOpts, varargin] = vl_argparse(runOpts, varargin);
            
    traxserver('setup', 'rectangle', {'path'});
    while true
        
        [image, region] = traxserver('wait');

        if isempty(image)
            break
        end

        img = read_img(image);

        if ~isempty(region)
            state = runOpts.state_initialize(img, region, runOpts);
            state = runOpts.initialize(state, img);
        else
            state = runOpts.track(state, img);
            state = runOpts.update(state);
        end

        if isempty(state.result(state.currFrame, :))
            state.result(state.currFrame, :) = [0, 0, 1, 1];
        end
        
        parameters = struct();
        parameters.confidence = state.targetScore;
        
        traxserver('status', double(state.result(state.currFrame, :)), parameters);
    end

    traxserver('quit');
    
catch err
    [wrapper_pathstr, ~, ~] = fileparts(mfilename('fullpath'));
    cd_ind = strfind(wrapper_pathstr, filesep());
    VOT_path = wrapper_pathstr(1:cd_ind(end));
    
    error_report_path = [VOT_path '/error_reports/'];
    if ~exist(error_report_path, 'dir')
        mkdir(error_report_path);
    end
    
    report_file_name = [error_report_path 'OptMDNet' datestr(now,'_yymmdd_HHMM') '.mat'];
    
    save(report_file_name, 'err')
    
    rethrow(err);
end


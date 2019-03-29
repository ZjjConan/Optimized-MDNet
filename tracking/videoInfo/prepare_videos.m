function [videos, cfgReader] = prepare_videos(dataset)

%%    
    if strcmpi(dataset, 'OTB2013')
        videos = OTB2013();
        cfgReader = @otb_info_loader;
    elseif strcmpi(dataset, 'OTB2015')
        videos = OTB2015();
        cfgReader = @otb_info_loader;
    end

end


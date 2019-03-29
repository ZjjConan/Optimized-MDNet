function imdb = mdnet_setup_imdb(varargin)
    
    opts.vidDir = '';
    opts.vidListDir = '';
    opts.vidList = '';
    opts.vidName = 'OTB';
    
    [opts, varargin] = vl_argparse(opts, varargin);
          

    for i = 1:numel(opts.vidListDir)
        v = textscan(...
                     fopen(...
                          fullfile(opts.vidListDir{i}, opts.vidList{i})), ...
                    '%s');
        videos{i} = v{1};
    end
    
    nset = numel(videos);
    
    imdb.images.set  = cell(nset, 1);
    imdb.images.fold = cell(nset, 1);
    imdb.images.bbox = cell(nset, 1);
    imdb.images.data = cell(nset, 1);
    imdb.images.imsz = cell(nset, 2);

    
    tic
    for s = 1:nset        
        imdb.images.set{s} = ones(numel(videos{s}), 1);
        imdb.images.fold{s} = ones(numel(videos{s}), 1);
        bbox = cell(numel(videos{s}), 1);
        data = cell(numel(videos{s}), 1);
        imsz = zeros(numel(videos{s}), 2);
        for v = 1:numel(videos{s})
            cfg = gen_config(opts.vidName, videos{s}{v}, opts.vidDir{s});
            data{v} = cfg.imgList';
            bbox{v}.gts = cfg.gt;
            iminfo = imfinfo(data{v}{1});
            imsz(v, :) = [iminfo.Height, iminfo.Width];
            
            fprintf('%s: process %s set %d / %d video time %.2fs\n', ...
                    mfilename, s, v, numel(videos{s}), toc); 
        end
        imdb.images.data{s} = data;
        imdb.images.bbox{s} = bbox;
        imdb.images.imsz{s} = imsz;
    end
    
    imdb.images.set = cat(1, imdb.images.set{:});
    imdb.images.fold = cat(1, imdb.images.fold{:});
    imdb.images.data = cat(1, imdb.images.data{:});
    imdb.images.bbox = cat(1, imdb.images.bbox{:});
    imdb.images.imsz = cat(1, imdb.images.imsz{:});
end


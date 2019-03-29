function setup_path_for_vot()

    [wrapper_path, name, ext] = fileparts(mfilename('fullpath'));

    addpath(wrapper_path)

    cd_ind = strfind(wrapper_path, filesep());
    repo_path = wrapper_path(1:cd_ind(end)-1);

    addpath(repo_path);

    setup_optmdnet();
end
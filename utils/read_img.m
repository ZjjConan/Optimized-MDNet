function img = read_img(imPath)
    img = imread(imPath);
    if size(img, 3) == 1
        img = repmat(img, [1 1 3]);
    end
end
function img = sub_average_img(img, opts)
    if ~isempty(opts.averageImage)
        if isscalar(opts.averageImage)
            img = img - opts.averageImage;
        else
            img = bsxfun(@minus, img, opts.averageImage);
        end
    end
end


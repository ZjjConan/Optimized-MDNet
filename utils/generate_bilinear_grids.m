function g = generate_bilinear_grids(p, s, opts)
    % bbox [x y w h]
    n = size(p, 2);
    p = p - 1;
    im_h = opts.imageSize(1) - 1;
    im_w = opts.imageSize(2) - 1;
    
    ty = p(2, :) * 2 / im_h - 1;
    tx = p(1, :) * 2 / im_w - 1;
    sy = s(2, :) ./ im_h;
    sx = s(1, :) ./ im_w;
    
    g = [sy; zeros(1, n, 'single'); zeros(1, n, 'single'); sx; ty; tx];
    g = reshape(g, 1, 1, size(g, 1), n);
    g = opts.gridGenerator.forward({g});
    g = g{1};
end

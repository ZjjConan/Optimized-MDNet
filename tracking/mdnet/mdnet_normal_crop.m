function ims = mdnet_normal_crop(im, boxes, opts)

im = gather(im);
numBoxes = size(boxes, 1);

cropMode = 'warp';
cropSize = opts.inputSize;
cropPadding = opts.cropPadding;

ims = zeros(cropSize, cropSize, 3, numBoxes, 'single');
% mean_rgb = mean(mean(single(im)));

for i = 1:numBoxes
    bbox = boxes(i,:);
    crop = im_crop(im, bbox, cropMode, cropSize, cropPadding);
    ims(:,:,:,i) = crop;
end
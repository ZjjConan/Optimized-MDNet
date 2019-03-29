function bbox = xyxy_to_xywh(bbox)
%XYWH_TO_XYXY 此处显示有关此函数的摘要
%   此处显示详细说明

    bbox(:, 3:4) = bbox(:, 3:4) - bbox(:, 1:2);

end


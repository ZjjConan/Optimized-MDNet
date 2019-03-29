function bbox = xywh_to_xyxy(bbox)
    bbox(:, 3:4) = bbox(:, 3:4) + bbox(:, 1:2);
end


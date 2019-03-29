function index = find_layer_index(net, layerName, functionHandle)
    index = find(functionHandle(@(a) strcmpi(a.name, layerName), net.layers) == 1);
end


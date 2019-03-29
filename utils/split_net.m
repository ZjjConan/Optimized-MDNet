function [net1, net2] = split_net(net, splitLayer)
    net1 = net.copy();
    net2 = net.copy();
    removedIndex = net1.getLayerIndex(splitLayer);
    layers = {net1.layers.name};    
    net1.removeLayer(layers(removedIndex:end));
    net2.removeLayer(layers(1:removedIndex-1));
end
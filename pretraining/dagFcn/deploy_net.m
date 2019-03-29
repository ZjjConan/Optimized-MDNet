
function net = deploy_net(net)
    net = dagnn.DagNN.loadobj(net);   

    dagRemoveLayersOfType(net, 'dagnn.Loss');
    dagRemoveLayersOfType(net, 'dagnn.ChnOut');

    lastConvSize = net.layers(end).block.size;
    lastConvSize(end) = 2;
    
    dagRemoveLayersOfName(net, net.layers(end).name);
    net = sort_layers(net);
    
    change_lr_wd(net, 'detconv1', [1 2], [1 0]);
    change_lr_wd(net, 'detconv2', [1 2], [1 0]);

    block = dagnn.Conv('size', lastConvSize);
    idx = net.getLayerIndex('detdrop2');
    if ~isnan(idx)
        net.addLayer('detconv3', block, net.layers(idx).outputs{1}, 'detconv3', {'detconv3f', 'detconv3b'});
    else
        net.addLayer('detconv3', block, net.layers(end).outputs{1}, 'detconv3', {'detconv3f', 'detconv3b'});
    end
    
    idx = find(arrayfun(@(a) strcmp(a.name, 'gpool'), net.layers) == 1); 
    if ~isempty(idx)
        net.setLayerInputs('gpool', {'detconv3'});
        net.setLayerOutputs('gpool', {'prediction'});
    else
        net.setLayerOutputs('detconv3', {'prediction'});
    end

    net.params(end-1).value        = 0.01 * randn(1, 1, lastConvSize(3), 2, 'single');
    net.params(end-1).learningRate = 1;
    net.params(end-1).weightDecay  = 1;
    net.params(end  ).value        = zeros(1, 2, 'single');
    net.params(end  ).learningRate = 2;
    net.params(end  ).weightDecay  = 0;
    
    net.addLayer('loss', dagnn.Loss(), {'prediction', 'label'}, 'loss');
end
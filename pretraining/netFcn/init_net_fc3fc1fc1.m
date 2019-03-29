function net = init_net_fc3fc1fc1(net, lastDim, numBranches)
    % Block 4
    block = dagnn.Conv('size', [3 3 lastDim 512], 'stride', 1, 'pad', 0);
    value = init_weights([3 3 lastDim 512]);
      
    net.addLayer('detconv1', block, net.getOutputs, 'detconv1', {'detconv1f', 'detconv1b'});
    net = assign_value(net, get_last_pindex(net), 'value', value);
    net = assign_value(net, get_last_pindex(net), 'learningRate', [10 20]);
    net = assign_value(net, get_last_pindex(net), 'weightDecay', [1 0]);  
    net.addLayer('detconv1x', dagnn.ReLU, 'detconv1', 'detconv1x');
    
    net.addLayer('detdrop1', dagnn.DropOut, 'detconv1x', 'detreg1');

    % Block 5
    block = dagnn.Conv('size', [1 1 512 512], 'stride', 1, 'pad', 0);
    value = init_weights([1 1 512 512]);
    
    net.addLayer('detconv2', block, net.getOutputs, 'detconv2', {'detconv2f', 'detconv2b'});
    net = assign_value(net, get_last_pindex(net), 'value', value);
    net = assign_value(net, get_last_pindex(net), 'learningRate', [10 20]);
    net = assign_value(net, get_last_pindex(net), 'weightDecay', [1 0]);
    
    net.addLayer('detconv2x', dagnn.ReLU, 'detconv2', 'detconv2x');
    
    net.addLayer('detdrop2', dagnn.DropOut, 'detconv2x', 'detreg2');

    % Block 6
    block = dagnn.Conv('size', [1 1 512 2*numBranches], 'stride', 1, 'pad', 0);
    value = init_weights([1 1 512  2*numBranches]);
    
    net.addLayer('detconv3', block, net.getOutputs, 'prediction', {'detconv3f', 'detconv3fb'});
    net = assign_value(net, get_last_pindex(net), 'value', value);
    net = assign_value(net, get_last_pindex(net), 'learningRate', [10 20]);
    net = assign_value(net, get_last_pindex(net), 'weightDecay', [1 0]);
end
function net = prepare_model(varargin)
    % conv1-3 layers from VGG-M network pretrained on ImageNet
    opts.initModel = 'D:/CNNModel/imagenet-vgg-m.mat';
    opts.isDagNN = false;
    opts.detNetType = 'fc3fc1fc1';
    opts.numBranches = 1;
    opts.removeAfterThisLayer = 'conv4';
    opts.usePad = false;
  
    [opts, varargin] = vl_argparse(opts, varargin);

    % load conv layers
    net = load(opts.initModel);
    if opts.isDagNN
        net = dagnn.DagNN.loadobj(net);
    else
        net = dagnn.DagNN.fromSimpleNN(net, 'CanonicalNames', true);
    end
    net.setLayerInputs(net.layers(1).name, {'input'});
    
    % change network structure
    pLayer = find_layer_index(net, opts.removeAfterThisLayer, @arrayfun);
    lname = {net.layers.name};
    lname = lname(pLayer:end);
    net.removeLayer(lname);
    numLayers = numel(net.layers);
   
    for i = 1:numLayers
        if isa(net.layers(i).block, 'dagnn.Conv')
            net.params(net.getParamIndex([net.layers(i).name 'f'])).learningRate = 1;
            net.params(net.getParamIndex([net.layers(i).name 'b'])).learningRate = 2;
            
            wsize = net.layers(i).block.size;
            
            if opts.usePad
                net.layers(i).block.pad = (wsize(1)-1)/2;
            else
                net.layers(i).block.pad = 0;
            end
                      
            lastDim = wsize(end);
            
        elseif isa(net.layers(i).block, 'dagnn.Pooling')
            wsize = net.layers(i).block.poolSize;
            if opts.usePad
                net.layers(i).block.pad = (wsize(1)-1)/2;
            else
                net.layers(i).block.pad = 0;
            end
        end
    end
    
    
    net = init_net_fc3fc1fc1(net, lastDim, opts.numBranches);
    pOut = net.layers(end).name;
         
    net.addLayer('all_loss', dagnn.SoftMaxKLoss('numBranches', opts.numBranches), ...
        {'prediction', 'k', 'label'}, 'all_loss');
        
    net.addLayer('pos_err', dagnn.MDPosError('numBranches', opts.numBranches), ...
        {'prediction', 'k', 'label'}, 'pos_err');

    net.addLayer('neg_err', dagnn.MDNegError('numBranches', opts.numBranches), ...
        {'prediction', 'k', 'label'}, 'neg_err');
    
    net.addLayer('all_err', dagnn.MDError('numBranches', opts.numBranches), ...
        {'prediction', 'k', 'label'}, 'all_err');
        
    net.setLayerOutputs(pOut, {'prediction'});
    net = sort_layers(net);
    net.rebuild();
end
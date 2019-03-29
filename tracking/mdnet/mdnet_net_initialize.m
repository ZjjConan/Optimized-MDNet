function [net_c, net_f] = mdnet_net_initialize(opts)

%%
    % --------
    % load net
    % --------
    net = dagnn.DagNN.loadobj(opts.netFile);
    net.setLayerInputs(net.layers(1).name, {'input'});
    net.setLayerOutputs('loss', {'loss'});
    net.vars(net.getVarIndex('prediction')).precious = 1;
    net.vars(net.getVarIndex('loss')).precious = 1;
    
    net = change_lr_wd(net, 'detconv1', [1 2], [1 0]);
    net = change_lr_wd(net, 'detconv2', [1 2], [1 0]);
    net = change_lr_wd(net, 'detconv3', [10 20], [1 0]);

    [net_c, net_f] = split_net(net, 'detconv1');
    net_f.setLayerInputs('detconv1', {'input'});
    net_f = sort_layers(net_f);
end
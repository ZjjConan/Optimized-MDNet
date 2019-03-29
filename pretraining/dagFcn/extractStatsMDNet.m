function stats = extractStatsMDNet(stats, net)
    sel = find(cellfun(@(x) isa(x,'dagnn.Loss'), {net.layers.block})) ;
    branch = net.vars(net.getVarIndex('k')).value;
    for i = 1:numel(sel)
        if net.layers(sel(i)).block.ignoreAverage, continue; end
        if ~isa(net.layers(sel(i)).block, 'dagnn.InsSoftMaxLoss')
            for b = 1:numel(branch)
                stats.(net.layers(sel(i)).outputs{1})(branch(b)) = net.layers(sel(i)).block.average(branch(b)) ;
            end
        else
            stats.(net.layers(sel(i)).outputs{1}) = net.layers(sel(i)).block.average;
        end
    end
end
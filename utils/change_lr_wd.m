function net = change_lr_wd(net, lname, lr, wd)
    pIndex = net.getParamIndex([lname 'f']);
    if ~isnan(pIndex)
        net.params(pIndex).learningRate = lr(1);
        net.params(pIndex).weightDecay = wd(1);
    end
    
    pIndex = net.getParamIndex([lname 'b']);
    if ~isnan(pIndex)
        net.params(pIndex).learningRate = lr(2);
        net.params(pIndex).weightDecay = wd(2);
    end
end
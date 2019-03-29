function dagRemoveLayersOfName(net, name)
% copy from matconvnet
names = dagFindLayersOfName(net, name) ;
for i = 1:numel(names)
  layer = net.layers(net.getLayerIndex(names{i})) ;
  net.removeLayer(names{i}) ;
  net.renameVar(layer.outputs{1}, layer.inputs{1}, 'quiet', true) ;
end
function layers = dagFindLayersOfName(net, name)
% copy from matconvnet
% layers = [] ;
layers = {net.layers.name};
layers = layers(find(strcmpi(layers, name) == 1));
% for l = 1:numel(net.layers)
%   if isa(net.layers(l).name, name)
%     layers{1,end+1} = net.layers(l).name ;
%   end
% end
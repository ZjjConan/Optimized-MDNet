classdef MDPosError < dagnn.Loss

  properties (Transient)
    numBranches = 1
  end

  methods
    function outputs = forward(obj, inputs, params)
      
      if size(inputs{1},3)/2 ~= numel(obj.average)  
          obj.numBranches = size(inputs{1},3)/2;
          obj.average = zeros(size(inputs{1},3)/2, 1);
          obj.numAveraged = zeros(size(inputs{1},3)/2, 1);
      end
        
      k = inputs{2};
      if(size(inputs{1},3)==2)
        predictions = gather(inputs{1}) ;
      else
        predictions = gather(inputs{1}(:,:,2*k-1:2*k,:)) ;
      end
     
      [~, predictions] = sort(predictions, 3, 'descend');
      sz = size(predictions);
      error = ~bsxfun(@eq, predictions, reshape(inputs{3}, 1, 1, 1, [])) ;
      idx = inputs{3} == 2;
      outputs{1} = sum(sum(sum(error(:,:,1,idx))))/prod(sz([1,2])); 
    
      % positive label = 2
      % negative label = 1
      n = obj.numAveraged(k);
      m = n + sum(idx) ;
  
      obj.average(k) = (n * obj.average(k) + gather(outputs{1})) / m ;
      obj.numAveraged(k) = m ;
      outputs{1} = obj.average(k);
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = [] ;
      derInputs{2} = [] ;
      derInputs{3} = [] ;
      derParams = {} ;
    end

    function reset(obj)
        % do nothing
        obj.average = obj.average;
        obj.numAveraged = obj.numAveraged;
    end

    function obj = MDPosError(varargin)
      obj.load(varargin) ;
      obj.numBranches = obj.numBranches;
      obj.average = zeros(obj.numBranches, 1);
      obj.numAveraged = zeros(obj.numBranches, 1);
    end
  end
end
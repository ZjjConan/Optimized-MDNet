classdef SoftMaxKLoss < dagnn.Loss
  
  properties
    numBranches = 1
  end
        
  methods
    function outputs = forward(obj, inputs, params)
      domains = inputs{2};
      batchSize = size(inputs{1}, 4) / numel(domains);
      for i = 1:numel(domains)
          k = domains(i);
          bstart = (i-1) * batchSize + 1;
          bend  = i * batchSize;
          batch = bstart:bend;
          outputs{1} = vl_nnloss(inputs{1}(:,:,2*k-1:2*k,batch), inputs{3}(batch), []) ;
          if numel(obj.numAveraged) ~= size(inputs{1},3)/2
            obj.numBranches = size(inputs{1},3)/2;
            obj.average = zeros(size(inputs{1},3)/2, 1);
            obj.numAveraged = zeros(size(inputs{1},3)/2, 1);
          end
          n = obj.numAveraged(k) ;
          m = n + numel(batch) ;
          obj.average(k) = (n * obj.average(k) + gather(outputs{1})) / m ;
          obj.numAveraged(k) = m ;
      end
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      domains = inputs{2};
      batchSize = size(inputs{1}, 4) / numel(domains);  
      for i = 1:numel(domains)  
          k = domains(i);
          bstart = (i-1) * batchSize + 1;
          bend  = i * batchSize;
          batch = bstart:bend;
          if isa(inputs{1}, 'gpuArray')
            derInputs{1} = gpuArray.zeros(size(inputs{1}), 'single');
          else
            derInputs{1} = zeros(size(inputs{1}), 'single');
          end
          derInputs{1}(:,:,2*k-1:2*k,batch) = vl_nnloss(inputs{1}(:,:,2*k-1:2*k,batch), inputs{3}(batch), derOutputs{1}) ;
      end
      derInputs{2} = [] ;
      derInputs{3} = [] ;
      derParams = {} ;
    end
    
    function reset(obj)
        % do nothing
        obj.average = obj.average;
        obj.numAveraged = obj.numAveraged;
    end
    
    function obj = SoftMaxKLoss(varargin)
      obj.load(varargin) ;
      obj.numBranches = obj.numBranches;
      obj.average = zeros(obj.numBranches, 1);
      obj.numAveraged = zeros(obj.numBranches, 1);
    end
  end
end

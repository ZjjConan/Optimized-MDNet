function state = mdnet_update(state)
    if((mod(state.currFrame, state.opts.updateInterval)==0 || state.targetScore < state.opts.updateThreshold))
        numSucess = numel(state.succIndex);
        if (state.targetScore < 0) % short-term update
            range = state.succIndex(max(1, numSucess - state.opts.numFramesST + 1) : numSucess);
        else % long-term update
            range = state.succIndex(max(1, numSucess - state.opts.numFramesLT + 1) : numSucess);
        end

        posData = cat(4, state.totalPosData{range});
        negData = cat(4, state.totalNegData{state.succIndex(max(1, end - state.opts.numFramesST + 1):end)});
        
        state.net_f = ...
            mdnet_finetune_hnm(state.net_f, posData, negData, state.opts, ...
                               'maxIters', state.opts.updateMaxIters, ...
                               'learningRate', state.opts.updateLr, ...
                               'batchSize4HNM', state.opts.updateBatchSize*8, ...
                               'batchSize', state.opts.updateBatchSize, ...
                               'batchPos', state.opts.updateBatchPos, ...
                               'batchNeg', state.opts.updateBatchNeg);
    end

end


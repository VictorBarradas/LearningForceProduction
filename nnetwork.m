classdef nnetwork
    %Simple two-layer neural network
    %
    
    properties
        nInput %Number of input units
        cInput %Centers of the input units
        omegaInput = 5; %Width of input units RBF
        
        nOutput %Number of output units
        W  %Weight matrix
    end
    
    methods
        function obj = nnetwork(nInput,nOutput)
            obj.nInput = nInput;
            obj.nOutput = nOutput;
            obj.cInput = -180:360/nInput:180-360/nInput;
            obj.W = zeros(nInput,nOutput);
        end
        
    end
end

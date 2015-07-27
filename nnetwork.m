classdef nnetwork < handle
    %Simple two-layer neural network
    %
    
    properties              
        omegaInput = 5; %Width of input units RBF
        sigmaInputUnits; %SD of noise in network inputs
        nInput %Number of input units
        nOutput %Number of output units
        cInput %Centers of the input units
        W  %Weight matrix
        type
    end
    
    methods (Abstract)
        network_feedforward(obj)
        network_learning(obj)
    end
    
    methods
        function obj = nnetwork(nInput,nOutput)
            obj.nInput = nInput;
            obj.nOutput = nOutput;
            obj.cInput = -180:360/nInput:180-360/nInput;
            obj.W = zeros(nInput,nOutput);
            obj.sigmaInputUnits = 0.0*ones(nInput,1);
        end
        
        function a = input_layer_activation(obj,direction,magnitude)
            % Input layer activation when presented to a target (RBF)
            a = exp(-log(2)*(angle_subtraction(direction, obj.cInput)/obj.omegaInput).^2); 
            
            % Noise injection
            inputNoise = normrnd(zeros(obj.nInput,1), obj.sigmaInputUnits);
            a = max(0, a + inputNoise);
        end
        
    end
    
   
end

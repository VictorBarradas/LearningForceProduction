classdef nnetwork < handle
    %Simple two-layer neural network
    %
    
    properties
        % These properties should be removed
        omegaInput = 5; %Width of input units RBF
        sigmaInputUnits; %SD of noise in network inputs
        cInput %Centers of the input units
        
        nLayers % Number of layers in the network, including input layer
        nUnits % Number of units in each layer, indicated by a vector
        nInput %Number of input units
        nOutput %Number of output units
        W  % Cell array of weight matrices
        type % type of neural network
    end
    
    methods (Abstract)
        network_feedforward(obj)
        network_learning(obj)
    end
    
    methods
        function obj = nnetwork(nUnits)
            obj.nUnits = nUnits;
            obj.nLayers = length(obj.nUnits);
            obj.nInput = obj.nUnits(1);
            obj.nOutput = obj.nUnits(end);
            obj.cInput = -180:360/obj.nInput:180-360/obj.nInput;
            for i = 1:obj.nLayers - 1
                obj.W{i} = zeros(obj.nUnits(i),obj.nUnits(i+1));
            end
            obj.sigmaInputUnits = 0.0*ones(obj.nInput,1);
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

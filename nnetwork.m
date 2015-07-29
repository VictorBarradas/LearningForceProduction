classdef nnetwork < handle
    %Simple two-layer neural network
    %
    
    properties
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
            for i = 1:obj.nLayers - 1
                obj.W{i} = zeros(obj.nUnits(i),obj.nUnits(i+1));
            end
        end
                
    end
     
end

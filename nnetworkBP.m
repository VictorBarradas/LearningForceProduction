classdef nnetworkBP < nnetwork
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties (Transient = true)
        alpha % learning rate
        activationOutput
        layerOutput
        layerInput
    end
    
    methods
        function obj = nnetworkBP(nUnits)
            obj@nnetwork(nUnits);
            for i = 1:obj.nLayers - 1
                obj.W{i} = normrnd(0,0.5,nUnits(i),nUnits(i+1));
            end
            obj.type = 'bp';
            obj.alpha = 0.025;
        end
        
        function neural_output = network_feedforward(obj,input)
            % Forward propagation (Input through output)
            obj.layerInput{1} = input;
            for i = 1:obj.nLayers - 1
                obj.layerOutput{i} = obj.W{i}'*obj.layerInput{i};
                obj.layerOutput{i} = 1./(1 + exp(-obj.layerOutput{i}));
                if i < obj.nLayers - 1
                    obj.layerInput{i+1} = obj.layerOutput{i};
                end
            end            
            neural_output = obj.layerOutput{end};
        end
        
        function network_learning(obj,error)
            % Backpropagation to weights between input and hidden layer
            deltaError = obj.layerOutput{end}.*(1-obj.layerOutput{end}).*error;
            tempW = obj.W{end};
            obj.W{end} = obj.W{end} + obj.alpha*obj.layerInput{end}*deltaError';
            for i = obj.nLayers-2:-1:1                          
                deltaError = obj.layerOutput{i}.*(1-obj.layerOutput{i}).*(tempW*deltaError);
                tempW = obj.W{i};
                obj.W{i} = obj.W{i} + obj.alpha*obj.layerInput{i}*deltaError';
            end
            
        end
    end
    
end


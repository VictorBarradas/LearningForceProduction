classdef nnetworkSRV < nnetwork
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        % Learning Properties
        V
        wThreshold
        vThreshold
        restrictExploration
    end
    
    properties (Transient = true)
        alpha % learning rate
        beta % learning rate
        layerInput
        expReward
        activationOutput
        muOutput
        sigmaOutput
        deltaW
    end
    
    methods
        function obj = nnetworkSRV(nUnits)
            obj@nnetwork(nUnits);
            for i = 1:obj.nLayers - 1
                obj.V{i} = zeros(obj.nUnits(i),obj.nUnits(i+1));
                obj.wThreshold{i} = zeros(obj.nUnits(i+1),1);
                obj.vThreshold{i} = zeros(obj.nUnits(i+1),1);
            end         
            obj.type = 'srv';
            % Learning
            obj.alpha = 0.005;
            obj.beta = 0.005;
            obj.restrictExploration = true;
        end
                
        function neural_output = network_feedforward(obj,input)
            obj.layerInput{1} = input;
            for i = 1:obj.nLayers - 1
                obj.muOutput{i} = obj.W{i}'*obj.layerInput{i} + obj.wThreshold{i};
                obj.expReward{i} = obj.V{i}'*obj.layerInput{i} + obj.vThreshold{i};
                %obj.sigmaOutput{i} = max(1 - obj.expReward{i}, 0.001);
                obj.sigmaOutput{i} = 2*exp(-(obj.expReward{i})/0.2);
                obj.activationOutput{i} = normrnd(obj.muOutput{i},obj.sigmaOutput{i});
                if obj.restrictExploration
                    b = obj.activationOutput{i} > obj.muOutput{i} + 2*obj.sigmaOutput{i};
                    obj.activationOutput{i}(b) = obj.muOutput{i}(b) + 2*obj.sigmaOutput{i}(b);
                    b = obj.activationOutput{i} < obj.muOutput{i} - 2*obj.sigmaOutput{i};
                    obj.activationOutput{i}(b) = obj.muOutput{i}(b) - 2*obj.sigmaOutput{i}(b);
                end
                if i < obj.nLayers - 1
                    obj.layerInput{i+1} = obj.activationOutput{i};
                end
            end
            neural_output = 1./(1 + exp(-obj.activationOutput{i}));
        end
        
        function network_learning(obj,reward)
            for i = obj.nLayers-1:-1:1
                obj.deltaW = (reward - obj.expReward{i}).*(obj.activationOutput{i} - obj.muOutput{i})./obj.sigmaOutput{i};
                weightTerm = obj.layerInput{i}*obj.deltaW';
                obj.W{i} = obj.W{i} + obj.alpha*weightTerm;
                obj.wThreshold{i} = obj.wThreshold{i} + obj.alpha*obj.deltaW;
                
                deltaV = reward - obj.expReward{i};
                obj.V{i} = obj.V{i} + obj.beta*obj.layerInput{i}*deltaV';
                obj.vThreshold{i} = obj.vThreshold{i} + obj.beta*deltaV;
            end
        end
    end
    
end


classdef nnetworkSRV < nnetwork
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        % Learning Properties
        V
        wThreshold
        vThreshold
    end
    
    properties (Transient = true)
        alpha % learning rate
        beta % learning rate
        input_activation
        expReward
        activationOutput
        muOutput
        sigmaOutput
    end
    
    methods
        function obj = nnetworkSRV(nInput,nOutput)
            obj@nnetwork(nInput,nOutput);
            obj.V = zeros(nInput, nOutput);
            obj.wThreshold = zeros(nOutput,1);
            obj.vThreshold = zeros(nOutput,1);
            obj.type = 'srv';
            % Learning
            obj.alpha = 0.005;
            obj.beta = 0.005;
        end
        
        function a = input_layer_activation(obj,direction,magnitude)
            a = input_layer_activation@nnetwork(obj,direction,magnitude);
        end
        
        function neural_output = network_feedforward(obj,direction,magnitude)
            obj.input_activation = input_layer_activation(obj,direction,magnitude);
            obj.muOutput = obj.W'*obj.input_activation + obj.wThreshold;
            obj.expReward = obj.V'*obj.input_activation + obj.vThreshold;
            obj.sigmaOutput = max(1 - obj.expReward, 0.01);
            obj.activationOutput = normrnd(obj.muOutput,obj.sigmaOutput);
            neural_output = 1./(1 + exp(-obj.activationOutput));
        end
        
        function network_learning(obj,reward)
            deltaW = (reward - obj.expReward).*(obj.activationOutput - obj.muOutput)./obj.sigmaOutput;
            weightTerm = obj.input_activation*deltaW';
            obj.W = obj.W + obj.alpha*weightTerm;
            obj.wThreshold = obj.wThreshold + obj.alpha*deltaW;
            
            deltaV = reward - obj.expReward;
            obj.V = obj.V + obj.beta*obj.input_activation*deltaV';
            obj.vThreshold = obj.vThreshold + obj.beta*deltaV;
        end
    end
    
end


classdef nnetworkAnnealing < nnetwork
    %Simple two-layer neural network
    %
    properties
        sigmaExplorationMax
        alpha
        gamma
    end
    properties (Transient = true)
        input_activation
        expReward
        exploration_noise
    end
        
    methods
        function obj = nnetworkAnnealing(nInput,nOutput)
            obj@nnetwork(nInput,nOutput);
            obj.sigmaExplorationMax = 20;
            obj.expReward = 0;
            obj.alpha = 0.025;
            obj.gamma = 0.0125;
            obj.type = 'anneal';
        end
        
        function a = input_layer_activation(obj,direction,magnitude)
            a = input_layer_activation@nnetwork(obj,direction,magnitude);
        end
        
        function neural_output = network_feedforward(obj,direction,magnitude,nTrials,currentTrial)
            sigmaExploration = obj.sigmaExplorationMax*(1 - obj.expReward)*(1 - currentTrial/nTrials)*ones(obj.nOutput,1);
            obj.exploration_noise = normrnd(0,sigmaExploration);
            obj.input_activation = input_layer_activation(obj,direction,magnitude);
            layer_output = obj.W'*obj.input_activation + obj.exploration_noise;
            neural_output = 1./(1 + exp(-1/10*(layer_output)));
        end
        
        function network_learning(obj,reward)
            obj.W = obj.W + obj.alpha*(reward - obj.expReward)*repmat(obj.input_activation,1,obj.nOutput).*repmat(obj.exploration_noise',obj.nInput,1);
            obj.expReward = obj.expReward + obj.gamma*(reward - obj.expReward);            
        end
        
    end
end

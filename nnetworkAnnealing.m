classdef nnetworkAnnealing < nnetwork
    %Simple two-layer neural network
    %
        
    methods
        function obj = nnetworkAnnealing(nInput,nOutput)
            obj@nnetwork(nInput,nOutput);
            obj.type = 'normal';
        end
        
        function a = input_layer_activation(obj,direction,magnitude)
            a = input_layer_activation@nnetwork(obj,direction,magnitude);
        end
        
        function [neural_output,input_activation] = network_feedforward(obj,target,exploration_noise)
            input_activation = input_layer_activation(obj,target);
            layer_output = obj.W'*input_activation + exploration_noise;
            neural_output = 1./(1 + exp(-1/10*(layer_output)));
        end
        
        function network_learning(obj)
        end
        
    end
end

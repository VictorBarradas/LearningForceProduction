classdef nnetworkSRV < nnetwork
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        V
        wThreshold
        vThreshold
    end
    
    methods
        function obj = nnetworkSRV(nInput,nOutput)
            obj@nnetwork(nInput,nOutput);
            obj.V = zeros(nInput, nOutput);
            obj.wThreshold = zeros(nOutput,1);
            obj.vThreshold = zeros(nOutput,1);
            obj.type = 'srv';
        end
        
        function a = input_layer_activation(obj,direction,magnitude)
            a = input_layer_activation@nnetwork(obj,direction,magnitude);
        end
        
        function [neural_output,input_activation,expReward,activationOutput,muOutput,sigmaOutput] = network_feedforward(obj,direction,magnitude)
            input_activation = input_layer_activation(obj,direction,magnitude);
            muOutput = obj.W'*input_activation + obj.wThreshold;
            expReward = obj.V'*input_activation + obj.vThreshold;
            sigmaOutput = max(1 - expReward, 0.01);
            activationOutput = normrnd(muOutput,sigmaOutput);
            neural_output = 1./(1 + exp(-activationOutput));
        end
        
        function network_learning(obj)
            
        end
    end
    
end


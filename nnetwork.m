classdef nnetwork < handle
    %Simple two-layer neural network
    %
    
    properties
        nInput %Number of input units
        cInput %Centers of the input units
        omegaInput = 5; %Width of input units RBF
        sigmaInputUnits; %SD of noise in network inputs
        nOutput %Number of output units
        W  %Weight matrix
        type = 'normal';
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
            a = magnitude*max(0, a + inputNoise);
        end
        
        function [neural_output,input_activation] = network_feedforward(obj,target,exploration_noise)
            input_activation = input_layer_activation(obj,target);
            layer_output = obj.W'*input_activation + exploration_noise;
            neural_output = 1./(1 + exp(-1/10*(layer_output)));
        end
        
    end
end

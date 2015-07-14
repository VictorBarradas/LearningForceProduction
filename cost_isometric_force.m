classdef cost_isometric_force < cost_function
    %UNTITLED3 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        weights
    end
    
    methods
        function obj = cost_isometric_force(weights)
            obj.weights = weights;
        end
        
        function u = cost(obj,errorForce,muscleActivation)
            u = obj.weights(1)*errorForce + obj.weights(2)*sum(muscleActivation.^2);
        end
    end
    
end


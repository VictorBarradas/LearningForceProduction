classdef cost_function < handle
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties (Abstract)
        weights
    end
    
    methods (Abstract)
        cost(obj)
    end
    
end


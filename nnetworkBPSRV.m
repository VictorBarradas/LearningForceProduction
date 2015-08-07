classdef nnetworkBPSRV < nnetwork
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        nnBP
        nnSRV
    end
    
    methods
        function obj = nnetworkBPSRV(nUnits)
            obj@nnetwork(nUnits);
            obj.nnBP = nnetworkBP(nUnits(1:end-1));
            obj.nnSRV = nnetworkSRV(nUnits(end-1:end));
            obj.type = 'bp_srv';
            
        end
        
        function neural_output = network_feedforward(obj,input)
            % Forward propagation (Input through last hidden)
            bpOutput = network_feedforward(obj.nnBP,input);
            neural_output = network_feedforward(obj.nnSRV,bpOutput);
        end
        
        function network_learning(obj,reward)
            tempW = obj.nnSRV.W{end};
            network_learning(obj.nnSRV,reward);
            errorBP = tempW*obj.nnSRV.deltaW;
            network_learning(obj.nnBP,errorBP);
        end
    end
    
end


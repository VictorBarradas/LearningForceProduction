classdef nnetworkBPSRV < nnetwork
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        V
        wThreshold
        vThreshold
        rewardThreshold
        restrictExploration
    end
    
    properties (Transient = true)
        alpha % learning rate
        beta % learning rate
        expReward
        activationOutput
        muOutput
        sigmaOutput
        hiddenOutput
        hiddenInput
    end
    
    methods
        function obj = nnetworkBPSRV(nUnits)
            obj@nnetwork(nUnits);
            for i = 1:obj.nLayers - 2
                obj.W{i} = normrnd(0,0.5,nUnits(i),nUnits(i+1));
            end
            obj.V = zeros(obj.nUnits(end-1),obj.nUnits(end));
            obj.wThreshold = zeros(obj.nUnits(end),1);
            obj.vThreshold = zeros(obj.nUnits(end),1);
            obj.type = 'bp_srv';
            obj.restrictExploration = true;
            obj.rewardThreshold = 2;
            obj.alpha = 0.025;
            obj.beta = 0.025;
        end
        
        function neural_output = network_feedforward(obj,input)
            % Forward propagation (Input through last hidden)
            obj.hiddenInput{1} = input;
            for i = 1:obj.nLayers - 2
                obj.hiddenOutput{i} = obj.W{i}'*obj.hiddenInput{i};
                obj.hiddenOutput{i} = 1./(1 + exp(-obj.hiddenOutput{i}));
                if i < obj.nLayers - 2
                    obj.hiddenInput{i+1} = obj.hiddenOutput{i};
                end
            end
            % Forward propagation (last hidden to Output)
            obj.muOutput = obj.W{end}'*obj.hiddenOutput{end} + obj.wThreshold;
            obj.expReward = obj.V'*obj.hiddenOutput{end} + obj.vThreshold;
            %obj.sigmaOutput = max((obj.rewardThreshold - obj.expReward), 0.001);
            obj.sigmaOutput = 2*exp(-(obj.expReward)/0.2);
            obj.activationOutput = normrnd(obj.muOutput,obj.sigmaOutput);
            if obj.restrictExploration
                b = obj.activationOutput > obj.muOutput + 1*obj.sigmaOutput;
                obj.activationOutput(b) = obj.muOutput(b) + 1*obj.sigmaOutput(b);
                b = obj.activationOutput < obj.muOutput - 1*obj.sigmaOutput;
                obj.activationOutput(b) = obj.muOutput(b) - 1*obj.sigmaOutput(b);
            end
            
            neural_output = 1./(1 + exp(-obj.activationOutput));
        end
        
        function network_learning(obj,reward)
            deltaW = (reward - obj.expReward).*(obj.activationOutput - obj.muOutput)./obj.sigmaOutput;
            weightTerm = obj.hiddenOutput{end}*deltaW';
            tempW = obj.W{end};
            obj.W{end} = obj.W{end} + obj.alpha*weightTerm;
            obj.wThreshold = obj.wThreshold + obj.alpha*deltaW;
            
            deltaV = reward - obj.expReward;
            obj.V = obj.V + obj.beta*obj.hiddenOutput{end}*deltaV';
            obj.vThreshold = obj.vThreshold + obj.beta*deltaV;
            
            % Backpropagation to weights between input and hidden layer
            deltaError = deltaW;
            for i = obj.nLayers-2:-1:1
            deltaError = obj.hiddenOutput{i}.*(1-obj.hiddenOutput{i}).*(tempW*deltaError);
            tempW = obj.W{i};
            obj.W{i} = obj.W{i} + obj.alpha*obj.hiddenInput{i}*deltaError';
            end
        end
    end
    
end


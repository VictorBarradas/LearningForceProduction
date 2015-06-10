classdef learning_framework
    %UNTITLED6 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        nn %Neural network object
        arm %Arm model object
    end
    
    methods
        function obj = learning_framework(nnetwork,arm)
            obj.nn = nnetwork;
            obj.arm = arm;
        end
        
        function train_force_annealing(obj,nTrainingGroup)
            % Reinforcement terms
            expReward = 0; % expected reward
            alpha = 0.025; % learning rate
            gamma = 0.0125; % learning rate
            rewardThreshold = 1;
            sigmaExplorationMax = 20; %max exploration noise
            nTrials = 1000;
            
            cTrainingGroup = -180:360/nTrainingGroup:180 - 360/nTrainingGroup;
            randomOrder = randperm(nTrainingGroup);
            bb = 1/8;
            cc = 1/2;
            desMagnitude = 4;
            
            for j = 1:nTrainingGroup
                desTheta = cTrainingGroup(randomOrder(j));
                for i = 1:nTrials
                    % Output
                    %Simulated annealing approach
                    sigmaExploration = sigmaExplorationMax*(1 - expReward)*(1 - i/nTrials)*ones(obj.nn.nOutput,1);
                    explorationNoise = normrnd(0,sigmaExploration);
                    
                    [muscleActivation,inputActivation] = network_feedforward(obj.nn,desTheta,explorationNoise);
                    
                    [magnitude,theta] = activation2force(obj.arm, muscleActivation);
                    
                    cost = (pi/180*(theta - desTheta))^2 + bb*(magnitude - desMagnitude)^2 + cc*sum(muscleActivation.^2);
                    
                    reward = max(0,(rewardThreshold - cost)/rewardThreshold); % reward function
                    %trackingVariable(i) = cost;
                    % Learning
                    
                    obj.nn.W = obj.nn.W + alpha*(reward - expReward)*repmat(inputActivation,1,obj.nn.nOutput).*repmat(explorationNoise',obj.nn.nInput,1);
                    expReward = expReward + gamma*(reward - expReward);
                    
                end
                %plot(trackingVariable);
            end
        end
        
        function plot_learned_force(obj,nPoints)
            cPoints = -180:360/nPoints:180 - 360/nPoints;
            for i=1:nPoints
                desTheta = cPoints(i);
                angle(i) = desTheta;
                inputActivation = input_layer_activation(obj.nn,desTheta);
                layerOutput = obj.nn.W'*inputActivation;
                muscleActivation = 1./(1 + exp(-1/10*(layerOutput)));
                [magnitude(i),theta(i)] = activation2force(obj.arm, muscleActivation);
            end
            
            plot(angle,theta,'.');
            title('Training positions')
            hold on
            plot(angle,angle)
            xlabel('Target direction')
            ylabel('Learned force direction');
            
            figure
            
            polar(cPoints*pi/180,magnitude);
            title('Learned force magnitude for every direction')
            hold on
        end
        
        function emg = muscle_activation(obj,nPoints)
            cPoints = -180:360/nPoints:180 - 360/nPoints;
            for i=1:nPoints
                desTheta = cPoints(i);
                exploration_noise = zeros(obj.nn.nOutput,1);
                emg(i,:) = network_feedforward(obj.nn,desTheta,exploration_noise);
            end
        end
        
        function plot_muscle_activations(obj,nPoints)
            emg = muscle_activation(obj,nPoints);
            cPoints = -180:360/nPoints:180 - 360/nPoints;
            for j = 1:obj.nn.nOutput
                figure
                polar(cPoints'*pi/180,ones(size(cPoints')),'k');
                hold on
                polar(cPoints'*pi/180,emg(:,j));
                title(obj.arm.muscle_names(j));
            end
        end
        
        function [WNorm,HNorm] = synergy_id(obj,nSynergies,nPoints)
            emg = muscle_activation(obj,nPoints);
            %Scale emg channels for unit variance
            stdev = std(emg');
            emgScaled = diag(1./stdev)*emg;
            [W,H] = nnmf(emgScaled,nSynergies);
            %Rescaling
            WRescaled = diag(stdev)*W
            %Synergy vector normalization
            m=max(WRescaled);% vector with max activation values
            for i=1:nSynergies
                HNorm(i,:)=H(i,:)*m(i);
                WNorm(:,i)=WRescaled(:,i)/m(i);
            end
        end
        
        %         function train_force_SRV(obj,nTrainingGroup)
        %
        %
        %         end
        
        
    end
    
end


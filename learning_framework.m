classdef learning_framework < handle
    %UNTITLED6 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        nn %Neural network object
        arm %Arm model object
        syn %Array with possible muscle synergies (according to nn.nOutput)
        nSyn
        emg % Learned EMG
    end
    
    methods
        function obj = learning_framework(nnetwork,arm)
            obj.nn = nnetwork;
            obj.arm = arm;
            obj.nSyn = nnetwork.nOutput;
            for i = 1:obj.nSyn
                temp(i) = synergy([],arm.muscle_names,i);
            end
            obj.syn = temp;
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
        
        function train_force_SRV(obj,nTrainingGroup,forceLevels)
            % Reinforcement terms
            alpha = 0.005; % learning rate
            beta = 0.005; % learning rate
            rewardThreshold = 1;
            nTrials = 3000;
            
            stackedForceLevels = repmat(forceLevels,nTrainingGroup,1);
            stackedForceLevels = stackedForceLevels(:);
            cTrainingGroup = repmat(-180:360/nTrainingGroup:180-360/nTrainingGroup,1,length(forceLevels));
            cTrainingGroup = [cTrainingGroup;stackedForceLevels'];
            randomOrder = randperm(length(forceLevels)*nTrainingGroup);
            aa = 1/8;
            bb = 1/8;
            cc = 1/8;
            
            for j = 1:nTrainingGroup*length(forceLevels)
                desTheta = cTrainingGroup(1,randomOrder(j));
                desMagnitude = cTrainingGroup(2,randomOrder(j));
                desForce = desMagnitude*[cos(desTheta*pi/180);sin(desTheta*pi/180)];
                for i = 1:nTrials
                    [muscleActivation,inputActivation,expReward,activationOutput,muOutput,sigmaOutput] = network_feedforward(obj.nn,desTheta,desMagnitude);
                    
                    endForce = activation2force(obj.arm, muscleActivation);
                    errorForce = norm(desForce - endForce);
                    
                    cost = aa*errorForce + cc*sum(muscleActivation.^2);
                    reward = max(0,(rewardThreshold - cost)/rewardThreshold)*ones(obj.nn.nOutput,1); % reward function
                    
                    % Learning
                    deltaW = (reward - expReward).*(activationOutput - muOutput)./sigmaOutput;
                    weightTerm = inputActivation*deltaW';
                    obj.nn.W = obj.nn.W + alpha*weightTerm;
                    obj.nn.wThreshold = obj.nn.wThreshold + alpha*deltaW;
                    
                    deltaV = reward - expReward;
                    obj.nn.V = obj.nn.V + beta*inputActivation*deltaV';
                    obj.nn.vThreshold = obj.nn.vThreshold + beta*deltaV;
                    
                end
            end
        end
        
        function plot_learned_force(obj,nPoints,forceLevels)            
            cPoints = -180:360/nPoints:180-360/nPoints;
            for j = 1:length(forceLevels)
                for i=1:nPoints
                    desTheta = cPoints(1,i);
                    desMagnitude = forceLevels(j);
                    angle(i) = desTheta;
                    if strcmp(obj.nn.type,'srv') == 1
                        muscleActivation = network_feedforward(obj.nn,desTheta,desMagnitude);
                    else
                        exploration_noise = zeros(obj.nn.nOutput,1);
                        muscleActivation = network_feedforward(obj.nn,desTheta,exploration_noise);
                    end
                    endForce = activation2force(obj.arm, muscleActivation);
                    theta(i) = 180/pi*atan2(endForce(2), endForce(1));
                    magnitude(i) = norm(endForce);  
                end
                
                figure
                
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
        end
        
        function plot_learning_error(obj,nPoints,desMagnitude)
            cPoints = -180:360/nPoints:180 - 360/nPoints;
            for i=1:nPoints
                desTheta = cPoints(i);
                angle(i) = desTheta;
                if strcmp(obj.nn.type,'srv') == 1
                    muscleActivation = network_feedforward(obj.nn,desTheta,desMagnitude);
                else
                    exploration_noise = zeros(obj.nn.nOutput,1);
                    muscleActivation = network_feedforward(obj.nn,desTheta,exploration_noise);
                end
                endForce = activation2force(obj.arm, muscleActivation);
                magnitude(i) = norm(endForce);
                theta(i) = 180/pi*atan2(endForce(2), endForce(1));
            end
            figure
            
            plot(angle,angle_subtraction(angle,theta));
            xlabel('Target direction');
            ylabel('Error in target direction')
            
            figure
            
            plot(angle,desMagnitude-magnitude);
            xlabel('Target direction');
            ylabel('Error in force magnitude');
        end
        
        function muscle_activation(obj,nPoints,desMagnitude)
            cPoints = -180:360/nPoints:180 - 360/nPoints;
            for i=1:nPoints
                desTheta = cPoints(i);
                if strcmp(obj.nn.type,'srv') == 1
                    obj.emg(:,i) = network_feedforward(obj.nn,desTheta,desMagnitude);
                else
                    noise = zeros(obj.nn.nOutput,1);
                    obj.emg(:,i) = network_feedforward(obj.nn,desTheta,noise);
                end
            end
        end
        
        function h = plot_muscle_activations(obj,nPoints,desMagnitude)
            if isempty(obj.emg)
                muscle_activation(obj,nPoints,desMagnitude);
            end
            cPoints = -180:360/nPoints:180 - 360/nPoints;
            for i = 1:obj.nn.nOutput
                h(i) = figure;
                polar(cPoints*pi/180,ones(size(cPoints)),'k');
                hold on
                polar(cPoints*pi/180,obj.emg(i,:));
                title(obj.arm.muscle_names(i));
            end
        end
        
        function plot_muscle_pulling_direction(obj,h)
            pullDir = pi/180*muscle_pulling_direction(obj.arm);
            for i = 1:obj.nn.nOutput
                figure(h(i));
                hold on
                polar([pullDir(i),pullDir(i)],[0,1],'r');
            end
        end
        
        function PD = muscle_preferred_direction(obj,nPoints,desMagnitude)
            if isempty(obj.emg)
                muscle_activation(obj,nPoints,desMagnitude);
            end
            cPoints = -180:360/nPoints:180 - 360/nPoints;
            cosfit = @(x,cPoints) x(1).*cosd(cPoints - x(2)) + x(3);
            guess = [1,90,0];
            lbound = [0,-180,0];
            ubound = [10,180,10];
            for i = 1:obj.nn.nOutput
                fit_values = lsqcurvefit(cosfit,guess,cPoints,obj.emg(i,:),lbound,ubound);
                PD(i) = fit_values(2);
            end
        end
        
        function identify_individual_synergy(obj,nPoints,nSynergy,desMagnitude)
            rEMG = muscle_activation(obj,nPoints,desMagnitude);
            obj.syn(nSynergy).rawEMG = rEMG;
            synergy_id(obj.syn(nSynergy));
            variance_within_synergies(obj.syn(nSynergy));
        end
        
        function identify_all_synergies(obj,nPoints,desMagnitude)
            for i = 1:obj.nSyn
                identify_individual_synergy(obj,nPoints,i,desMagnitude);
            end
        end
        
        function plot_found_synergies(obj)
            for i = 1:obj.nSyn
                plot_synergy(obj.syn(i));
            end
        end
        
        function plot_rr_curve(obj)
            for i = 1:obj.nSyn
                rr(i) = rsquared(obj.syn(i));
            end
            figure
            plot(rr);
            xlabel('Number of synergies');
            ylabel('Corr Coef');
            ylim([0 1]);
            set(gca,'XTick',1:1:obj.nSyn);
            title('Goodness of fit');
        end
        
        function plot_vaf_curve(obj)
            for i = 1:obj.nSyn
                vaf(i) = vaf_fit(obj.syn(i));
            end
            figure
            plot(vaf);
            xlabel('Number of synergies');
            ylabel('VAF [%]');
            ylim([0 100]);
            set(gca,'XTick',1:1:obj.nSyn);
            title('Goodness of fit');
        end
        
        function plot_vaf_muscle_curve(obj)
            figure
            color_code = hsv(obj.nSyn);
            for i = 1:obj.nSyn
                vaf_muscle = vaf_fit_muscles(obj.syn(i));
                plot(vaf_muscle,'color',color_code(i,:));
                hold on
            end
            xlabel('Number of synergies');
            ylabel('VAF [%]');
            ylim([0 100]);
            set(gca,'XTick',1:1:obj.nSyn);
            title('Goodness of fit for individual muscles');
            legend(obj.arm.muscle_names);
        end
    end
    
end


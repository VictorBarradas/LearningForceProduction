classdef learning_framework < handle
    %UNTITLED6 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        nn % Neural network object
        arm % Arm model object
        syn % Array with possible muscle synergies (according to nn.nOutput)
        nSyn
        emg % Learned EMG
        force_levels % Magnitudes of forces to be learned by the system
        cost_function % Cost function object
    end
    
    methods
        function obj = learning_framework(nnetwork,arm,costF)
            obj.nn = nnetwork;
            obj.arm = arm;
            obj.cost_function = costF;
            obj.nSyn = nnetwork.nOutput;
            for i = 1:obj.nSyn
                temp(i) = synergy([],arm.muscle_names,i);
            end
            obj.syn = temp;
            obj.force_levels = [];
        end
        
        function train_force_production(obj,nTrainingGroup,forceLevels)
            obj.force_levels = forceLevels;
            
            nTrials = 3000;
            
            stackedForceLevels = repmat(obj.force_levels,nTrainingGroup,1);
            stackedForceLevels = stackedForceLevels(:);
            cTrainingGroup = repmat(-180:360/nTrainingGroup:180-360/nTrainingGroup,1,length(obj.force_levels));
            cTrainingGroup = [cTrainingGroup;stackedForceLevels'];
            randomOrder = randperm(length(obj.force_levels)*nTrainingGroup);
            rewardThreshold = 1; % threshold for reward
            
            for j = 1:nTrainingGroup*length(obj.force_levels)
                desTheta = cTrainingGroup(1,randomOrder(j));
                desMagnitude = cTrainingGroup(2,randomOrder(j));
                desForce = desMagnitude*[cos(desTheta*pi/180);sin(desTheta*pi/180)];
                for i = 1:nTrials
                    if strcmp(obj.nn.type,'anneal')
                         muscleActivation = network_feedforward(obj.nn,desTheta,desMagnitude,nTrials,i);
                    elseif strcmp(obj.nn.type,'srv')
                        muscleActivation = network_feedforward(obj.nn,desTheta,desMagnitude);
                    end
                    
                    % Interaction with the environment
                    endForce = activation2force(obj.arm, muscleActivation);
                    errorForce = norm(desForce - endForce);
                    % Cost and evaluation
                    u = cost(obj.cost_function,errorForce,muscleActivation);
                    
                    if strcmp(obj.nn.type,'anneal')
                        reward = max(0,(rewardThreshold - u)/rewardThreshold);
                    elseif strcmp(obj.nn.type,'srv')
                        reward = max(0,(rewardThreshold - u)/rewardThreshold)*ones(obj.nn.nOutput,1); % reward function
                    end
                    network_learning(obj.nn,reward);                   
                end
            end
        end
        
        function plot_learned_force(obj,nPoints)            
            cPoints = -180:360/nPoints:180-360/nPoints;
            for j = 1:length(obj.force_levels)
                for i=1:nPoints
                    desTheta = cPoints(1,i);
                    desMagnitude = obj.force_levels(j);
                    angle(i) = desTheta;
                    if strcmp(obj.nn.type,'srv') == 1
                        muscleActivation = network_feedforward(obj.nn,desTheta,desMagnitude);
                    elseif strcmp(obj.nn.type,'anneal') == 1
                        muscleActivation = network_feedforward(obj.nn,desTheta,desMagnitude,1,1);
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
        
        % Modify function to admit several force levels
        function plot_learning_error(obj,nPoints)
            desMagnitude = obj.force_levels;
            cPoints = -180:360/nPoints:180 - 360/nPoints;
            for i=1:nPoints
                desTheta = cPoints(i);
                angle(i) = desTheta;
                if strcmp(obj.nn.type,'srv') == 1
                    muscleActivation = network_feedforward(obj.nn,desTheta,desMagnitude);
                elseif strcmp(obj.nn.type,'anneal') == 1
                    muscleActivation = network_feedforward(obj.nn,desTheta,desMagnitude,1,1);
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
        
        % Modify function to admit several force levels
        function muscle_activation(obj,nPoints)
            desMagnitude = obj.force_levels;
            cPoints = -180:360/nPoints:180 - 360/nPoints;
            for i=1:nPoints
                desTheta = cPoints(i);
                if strcmp(obj.nn.type,'srv') == 1
                    obj.emg(:,i) = network_feedforward(obj.nn,desTheta,desMagnitude);
                elseif strcmp(obj.nn.type,'anneal')
                    obj.emg(:,i) = network_feedforward(obj.nn,desTheta,desMagnitude,1,1);
                end
            end
        end
        
        function h = plot_muscle_activations(obj,nPoints)
            if isempty(obj.emg)
                muscle_activation(obj,nPoints);
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
        
        function PD = muscle_preferred_direction(obj,nPoints)
            if isempty(obj.emg)
                muscle_activation(obj,nPoints);
            end
            cPoints = -180:360/nPoints:180 - 360/nPoints;
            cosfit = @(x,cPoints) x(1).*cosd(cPoints - x(2)) + x(3);
            guess = [1,90,0];
            lbound = [0,-270,0];
            ubound = [10,270,10];
            for i = 1:obj.nn.nOutput
                fit_values = lsqcurvefit(cosfit,guess,cPoints,obj.emg(i,:),lbound,ubound);
                PD(i) = fit_values(2);
            end
        end
        
        function plot_muscle_preferred_direction(obj,nPoints,h)
            PD = pi/180*muscle_preferred_direction(obj,nPoints);
            for i = 1:obj.nn.nOutput
                figure(h(i));
                hold on
                polar([PD(i),PD(i)],[0,1],'k');
            end
        end
        
        function identify_individual_synergy(obj,nPoints,nSynergy)
            if isempty(obj.emg)
                muscle_activation(obj,nPoints);
            end
            obj.syn(nSynergy).rawEMG = obj.emg;
            synergy_id(obj.syn(nSynergy));
            variance_within_synergies(obj.syn(nSynergy));
        end
        
        function identify_all_synergies(obj,nPoints)
            for i = 1:obj.nSyn
                identify_individual_synergy(obj,nPoints,i);
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


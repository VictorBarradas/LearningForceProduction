function [W] = learn_force_production(arm_physics)
% Reinforcement learning for force production of a two joint planar arm
% theta: direction of desired force (degrees)
% For now: count only direction

%close all

% Input layer: neuron population for values of x around a circle (-180,180)

omegaInput = 5; % Assume all input units have the same width
nInput = 180; % Number of input units
cInput = -180:360/nInput:180-360/nInput; % centers of the input units

sigmaNoise = 0.0*ones(nInput,1);

nOutput = 6;

% Connectivity (weight) matrix (W)
W = zeros(nInput, nOutput);

% Arm position
armPosition = [45, 90];

% Reinforcement terms
expReward = 0; % expected reward
alpha = 0.025; % learning rate
gamma = 0.0125; % learning rate
rewardThreshold = 1;
sigmaExplorationMax = 20; %max exploration noise
nTrials = 1000;

nTrainingGroup = 20;
cTrainingGroup = -180:360/nTrainingGroup:180 - 360/nTrainingGroup;
randomOrder = randperm(nTrainingGroup);
bb = 1/8;
cc = 1/2;
desMagnitude = 4;

for j = 1:nTrainingGroup
    desTheta = cTrainingGroup(randomOrder(j));
    for i = 1:nTrials
        % Activation layer
        
        inputActivation = exp(-log(2)*(angle_subtraction(desTheta, cInput)/omegaInput).^2); %RBF
        
        % Noise injection
        inputNoise = normrnd(zeros(nInput,1), sigmaNoise);
        inputActivation = max(0, inputActivation + inputNoise);
        
        % Output
        sigmaExploration = sigmaExplorationMax*(1 - expReward)*(1 - i/nTrials)*ones(nOutput,1); %Simulated annealing approach
        explorationNoise = normrnd(0,sigmaExploration);
        layerOutput = W'*inputActivation + explorationNoise;

        muscleActivation = 1./(1 + exp(-1/10*(layerOutput)));

        producedForce = arm_physics(armPosition, muscleActivation);
        producedMagnitude = sqrt(producedForce(1)^2 + producedForce(2)^2);
        producedTheta = 180/pi*atan2(producedForce(1), producedForce(2));
        
        cost = (pi/180*(producedTheta - desTheta))^2 + bb*(producedMagnitude - desMagnitude)^2 + cc*sum(muscleActivation.^2);
        
        reward = max(0,(rewardThreshold - cost)/rewardThreshold); % reward function
        trackingVariable(i) = cost;
        % Learning
        
        W = W + alpha*(reward - expReward)*repmat(inputActivation,1,nOutput).*repmat(explorationNoise',nInput,1);
        expReward = expReward + gamma*(reward - expReward);
        
    end
    %plot(trackingVariable);
end

% for i=1:nTrainingGroup
%     desTheta = cTrainingGroup(randomOrder(i));
%     angle(i) = desTheta;
%     inputActivation = exp(-log(2)*(angle_subtraction(desTheta, cInput)/omegaInput).^2); %RBF
%     inputNoise = max(0, normrnd(zeros(nInput,1), sigmaNoise));
%     inputActivation = inputActivation + inputNoise;
%     layerOutput = W'*inputActivation;
%     muscleActivation = 1./(1 + exp(-1/10*(layerOutput)));
%     producedForce = arm_physics(armPosition, muscleActivation);
%     producedMagnitude(i) = sqrt(producedForce(1)^2 + producedForce(2)^2);
%     producedTheta(i) = 180/pi*atan2(producedForce(1), producedForce(2));
% end
% 
% 
% plot(angle,producedTheta,'.');
% title('Training positions')
% hold on
% plot(angle,angle)
% xlabel('Target direction')
% ylabel('Learned force direction');
% 
% figure
% 
% polar(cTrainingGroup*pi/180,producedMagnitude);
% title('Learned force magnitude for every direction')
% hold on
% 
% 
% 
% for i=1:nInput
%     desTheta = cInput(i);
%     inputActivation = exp(-log(2)*(angle_subtraction(desTheta, cInput)/omegaInput).^2); %RBF
%     layerOutput = W'*inputActivation;
%     muscleActivations(i,:) = 1./(1 + exp(-1/10*(layerOutput)));
% end
% 
% figure
% subplot(1,3,1);
% polar(cInput'*pi/180,ones(size(cInput')),'k');
% hold on
% polar(cInput'*pi/180,muscleActivations(:,1));
% title('Shoulder activations');
% polar(cInput'*pi/180,muscleActivations(:,2),'r');
% 
% subplot(1,3,2);
% polar(cInput'*pi/180,ones(size(cInput')),'k');
% hold on
% polar(cInput'*pi/180,muscleActivations(:,3));
% title('Elbow activations');
% polar(cInput'*pi/180,muscleActivations(:,4),'r');
% 
% subplot(1,3,3);
% polar(cInput'*pi/180,ones(size(cInput')),'k');
% hold on
% polar(cInput'*pi/180,muscleActivations(:,5));
% title('Biarticular activations');
% polar(cInput'*pi/180,muscleActivations(:,6),'r');
% 
% figure
% 
% subplot(6,1,1);
% colormap hot
% contourf(repmat(transpose(W(:,1)),2,1),'LineStyle','None');
% title('Shoulder Flexor Weights')
% set(gca,'YTickLabel',[]);
% 
% subplot(6,1,2);
% colormap hot
% contourf(repmat(transpose(W(:,2)),2,1),'LineStyle','None');
% title('Shoulder Extensor Weights')
% set(gca,'YTickLabel',[]);
% 
% subplot(6,1,3);
% colormap hot
% contourf(repmat(transpose(W(:,3)),2,1),'LineStyle','None');
% title('Elbow Flexor Weights')
% set(gca,'YTickLabel',[]);
% 
% subplot(6,1,4);
% colormap hot
% contourf(repmat(transpose(W(:,4)),2,1),'LineStyle','None');
% title('Elbow Extensor weights')
% set(gca,'YTickLabel',[]);
% 
% subplot(6,1,5);
% colormap hot
% contourf(repmat(transpose(W(:,5)),2,1),'LineStyle','None');
% title('Biarticular 1 Weights')
% set(gca,'YTickLabel',[]);
% 
% subplot(6,1,6);
% contourf(repmat(transpose(W(:,6)),2,1),'LineStyle','None');
% title('Biarticular 2 Weights');
% set(gca,'YTickLabel',[]);

end


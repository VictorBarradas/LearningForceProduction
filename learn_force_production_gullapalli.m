function [] = learn_force_production_gullapalli(arm_physics)
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
V = zeros(nInput, nOutput);

wThreshold = zeros(nOutput,1);
vThreshold = zeros(nOutput,1);

% Arm position
armPosition = [0, 90];

% Reinforcement terms
alpha = 0.005; % learning rate
beta = 0.005; % learning rate
rewardThreshold = 1;
nTrials = 3000;

nTrainingGroup = 180;
cTrainingGroup = -180:360/nTrainingGroup:180-360/nTrainingGroup;
randomOrder = randperm(nTrainingGroup);
magnitudeCostWeight = 1/8;
muscActCostWeight = 1/2;
desMagnitude = 5;

for j = 1:nTrainingGroup
    desTheta = cTrainingGroup(randomOrder(j));
    for i = 1:nTrials
        % Activation layer
        
        inputActivation = exp(-log(2)*(angle_subtraction(desTheta, cInput)/omegaInput).^2); %RBF
        
        % Noise injection
        inputNoise = normrnd(zeros(nInput,1), sigmaNoise);
        inputActivation = max(0, inputActivation + inputNoise);
        
        % Output
        muOutput = W'*inputActivation + wThreshold;
        expReward = V'*inputActivation + vThreshold;
        sigmaOutput = max(1 - expReward, 0.01);
        activationOutput = normrnd(muOutput, sigmaOutput);
        
        muscleActivation = 1./(1 + exp(-activationOutput));

        producedForce = arm_physics(armPosition, muscleActivation);
        producedMagnitude = sqrt(producedForce(1)^2 + producedForce(2)^2);
        producedTheta = 180/pi*atan2(producedForce(1), producedForce(2));
        
        cost = (pi/180*(producedTheta - desTheta))^2 + magnitudeCostWeight*(producedMagnitude - desMagnitude)^2 + muscActCostWeight*sum(muscleActivation.^2);
        reward = max(0,(rewardThreshold - cost)/rewardThreshold)*ones(nOutput,1); % reward function
        trackingVariable(i) = cost;
        
        % Learning
        deltaW = (reward - expReward).*(activationOutput - muOutput)./sigmaOutput;
        weightTerm = inputActivation*deltaW';
        W = W + alpha*weightTerm;
        wThreshold = wThreshold + alpha*deltaW;
        
        deltaV = reward - expReward;
        V = V + beta*inputActivation*deltaV';
        vThreshold = vThreshold + beta*deltaV;
        
    end
    %plot(trackingVariable);
end

for i=1:nTrainingGroup
    desTheta = cTrainingGroup(randomOrder(i));
    angle(i) = desTheta;
    inputActivation = exp(-log(2)*(angle_subtraction(desTheta, cInput)/omegaInput).^2); %RBF
    inputNoise = max(0, normrnd(zeros(nInput,1), sigmaNoise));
    inputActivation = inputActivation + inputNoise;
    
    muOutput = W'*inputActivation + wThreshold;
    expReward = V'*inputActivation + vThreshold;
    sigmaOutput = max(1 - expReward, 0.01);
    activationOutput = normrnd(muOutput, sigmaOutput);
    
    muscleActivation = 1./(1 + exp(-activationOutput));

    producedForce = arm_physics(armPosition, muscleActivation);
    producedMagnitude(i) = sqrt(producedForce(1)^2 + producedForce(2)^2);
    producedTheta(i) = 180/pi*atan2(producedForce(1), producedForce(2));
end

plot(angle,producedTheta,'.');
title('Training positions')
hold on
plot(angle,angle)
xlabel('Target direction')
ylabel('Learned force direction');

figure

polar(cTrainingGroup*pi/180,producedMagnitude);
title('Learned force magnitude for every direction')
hold on

for i=1:nInput
    desTheta = cInput(i);
    inputActivation = exp(-log(2)*(angle_subtraction(desTheta, cInput)/omegaInput).^2); %RBF
    inputNoise = max(0, normrnd(zeros(nInput,1), sigmaNoise));
    inputActivation = inputActivation + inputNoise;
    
    muOutput = W'*inputActivation + wThreshold;
    %expReward = V'*inputActivation + vThreshold;
    %sigmaOutput = max(1 - expReward, 0.01);
    %activationOutput = normrnd(muOutput, sigmaOutput);
    
    muscleActivation(:,i) = 1./(1 + exp(-muOutput));
end

figure
subplot(1,3,1);
h0 = polar(cInput*pi/180,ones(size(cInput)));
hold on
h1 = polar(cInput*pi/180,muscleActivation(1,:));
%title('Shoulder activations');
h2 = polar(cInput*pi/180,muscleActivation(2,:));
set(gca,'xtick',[]);
set(h0,'visible','off');
set(h1,'color','b','linewidth',2);
set(h2,'color','r','linewidth',2);

subplot(1,3,2);
h00 = polar(cInput*pi/180,ones(size(cInput)),'k');
hold on
h3 = polar(cInput*pi/180,muscleActivation(3,:));
%title('Elbow activations');
h4 = polar(cInput*pi/180,muscleActivation(4,:));
set(h00,'visible','off');
set(h3,'color','b','linewidth',2);
set(h4,'color','r','linewidth',2);

subplot(1,3,3);
h000 = polar(cInput*pi/180,ones(size(cInput)),'k');
hold on
h5 = polar(cInput*pi/180,muscleActivation(5,:));
%title('Biarticular activations');
h6 = polar(cInput*pi/180,muscleActivation(6,:));
set(h000,'visible','off');
set(h5,'color','b','linewidth',2);
set(h6,'color','r','linewidth',2);

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
% 
% end


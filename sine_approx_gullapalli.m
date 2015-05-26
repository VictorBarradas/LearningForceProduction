function [] = sine_approx_gullapalli()
% Reinforcement learning for force production of a two joint planar arm
% theta: direction of desired force (degrees)
% For now: count only direction

close all

% Input layer: neuron population for values of x around a circle (-180,180)

omegaInput = 0.1; % Assume all input units have the same width
nInput = 20; % Number of input units
cInput = 0:1/(nInput-1):1; % centers of the input units

sigmaNoise = 0.0*ones(nInput,1);

nOutput = 6;

% Connectivity (weight) matrix (W)
W = zeros(nInput, nOutput);
V = zeros(nInput, nOutput);

wThreshold = zeros(nOutput,1);
vThreshold = zeros(nOutput,1);

% Reinforcement terms
alpha = 0.02; % learning rate
beta = 0.02; % learning rate
rewardThreshold = 0.1;
nTrials = 500;

nTrainingGroup = 180;
cTrainingGroup = 0:1/(nTrainingGroup-1):1;
randomOrder = randperm(nTrainingGroup);

for j = 1:nTrainingGroup
    xPos = cTrainingGroup(randomOrder(j));
    desFunction = sin(2*pi*xPos);
    for i = 1:nTrials
        % Activation layer
        
        inputActivation = exp(-log(2)*((xPos - cInput)/omegaInput).^2); %RBF
        inputActivation = inputActivation';
        
        % Noise injection
        inputNoise = normrnd(zeros(nInput,1), sigmaNoise);
        inputActivation = max(0, inputActivation + inputNoise);
        
        % Output
        muOutput = W'*inputActivation + wThreshold;
        expReward = V'*inputActivation + vThreshold;
        sigmaOutput = max(0.5 - expReward, 0.01);
        activationOutput = normrnd(muOutput, sigmaOutput);
        unitOutput = sum(activationOutput);
        
        cost = (unitOutput - desFunction).^2;
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
    %     figure
    %     plot(trackingVariable);
end
for k = 1:nInput
    x = cInput(k);
    inputActivation = exp(-log(2)*((x - cInput)/omegaInput).^2); %RBF
    inputActivation = inputActivation';
    
    muOutput = W'*inputActivation + wThreshold;
    expReward = V'*inputActivation + vThreshold;
    sigmaOutput = max(0.5 - expReward, 0.01);
    activationOutput = normrnd(muOutput, sigmaOutput);
    unitOutput = sum(activationOutput);
    z(k) = unitOutput;
end
zzz = sin(2*pi*cInput);
figure
plot(cInput,z)
hold on
plot(cInput,zzz,'r')
%plot(trackingVariable);
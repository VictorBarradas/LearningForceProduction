function [] = sine_approx()
% Function to approximate = sin(2*pi*t)
%   Detailed explanation goes here

% Input layer: neuron population for values of xE[0,1]

omega = 0.1; % Assume all input units have the same width
nInput = 20; % Number of input units
c = 0:1/nInput:1; % centers of the input units
sigmaNoise = 0.2*ones(1, nInput + 1);

% Input to the network
z = [0.25, 0.75];

% Connectivity (weight) matrix (W)
W = zeros(nInput + 1, 1);rand(nInput + 1, 1);

% Reinforcement terms
rExp = 0; % expected reward
alpha = 0.05; % learning rate
gamma = 0.05; % learning rate
threshold = 3;
sigmaMax = 2;
numTrials = 500;

for j = 1:nInput
    x = c(j);
    desFunction = sin(2*pi*x);
    for i = 1:numTrials
        % Activation layer
        a = exp(-log(2)*((x - c)/omega).^2); %RBF
        
        % Noise injection
        noise = max(0, normrnd(zeros(1,nInput + 1), sigmaNoise));
        a = a + noise;
        a = a';
        
        % Output
        sigmaExploration = sigmaMax*(1 - rExp)*(1 - i/numTrials);
        explorationNoise = normrnd(0,sigmaExploration);
        y = W'*a + explorationNoise;
        
        
        cost = (y - desFunction).^2;
        r = max(0,(threshold - cost)/threshold); % reward function
        
        % Learning
        
        W = W + alpha*(r - rExp)*a*explorationNoise;
        rExp = rExp + gamma*(r - rExp);
        
        storyCost(i) = y;
    end
end

plot(storyCost);

for i = 1:nInput + 1
    x = c(i);
    a = exp(-log(2)*((x - c)/omega).^2); %RBF
    
    % Noise injection
    noise = max(0, normrnd(zeros(1,nInput + 1), sigmaNoise));
    a = a + noise;
    a = a';
    y(i) = W'*a;
end

desFunction = sin(2*pi*c);
plot(c,y)
hold on
plot(c,desFunction,'r')

end


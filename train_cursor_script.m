close all
clc
clear
% Test script for training arm to move cursor in virtual environment

% Initialize arm
arm = arm_model([45;90],0,0,0,0,0);
default_six_muscles(arm);

% Create neural network:
% Input layer: 2 inputs: x and y positions in plane
% Hidden layer: N backpropagation units
% Output layer: M SRV units (M is number of muscles in model)

nInput = 2;
nHidden = 50;
nOutput = size(arm.R,2);

nn = nnetworkBPSRV([nInput,nHidden,nOutput]);

% Task parameters
nTrials = 200;
nAttempts = 500;
parameters = [1;1;1];
timestep = 0.05;
rewardThreshold = 2;
alpha = 0.025;
beta = 0.025;

% Start training
for i = 1: nAttempts
    target = [-1;0];
    % Virtual environment
   
    %plot_target(ve,1,target);
    rho = 3*rand;
    phi = rand*2*pi;
    input_vector = rho*[cos(phi);sin(phi)];
    initial_state = [input_vector;0;0];
    ve = virtual_environment(initial_state,parameters);
    for j = 1:nTrials
        muscleActivation = network_feedforward(nn,input_vector);
        
        % Interaction with the environment
        endForce = activation2force(arm, muscleActivation);
        forward_simulation(ve,endForce,timestep);
        %         plot_cursor(ve,1);
        %         pause(0.01);
        pos_output = ve.position;
        
        % Cost and evaluation
        pos_error = norm(target - pos_output);
        cost = pos_error + 1/128*sum(muscleActivation.^2);
        reward = max(0,(rewardThreshold - cost)/rewardThreshold);
        temp(j) = reward;

        network_learning(nn,reward);
        
        input_vector = pos_output;
    end
    %     figure(2)
    %plot(temp);
end

%Testing the learning

for i = 1:10
    % Virtual environment
    initial_state = [0;0;0;0];
    ve = virtual_environment(initial_state,parameters);
    plot_target(ve,1,target);
    input_vector = [0;0];
    for j = 1:nTrials
        muscleActivation = network_feedforward(nn,input_vector);
        
        % Interaction with the environment
        endForce = activation2force(arm, muscleActivation);
        forward_simulation(ve,endForce,timestep);
        pos_output = ve.position;
        plot_cursor(ve,1);
        pause(0.01);
        input_vector = pos_output;
    end
    close(1)
end







%% Synergy extraction from reinforcement learning of force production in an arm
close all
clc
clear
arm = arm_model([45;90],0,0,0,0,0);
default_six_muscles(arm);

%for j = 1:4
nn = nnetwork(180,size(arm.R,2));
learn_process = learning_framework(nn,arm);
train_force_annealing(learn_process,200);

%% Learned task
plot_learned_force(learn_process,200)
%plot_muscle_activations(learn_process,200);

%% Identified synergies
identify_all_synergies(learn_process,200);
plot_found_synergies(learn_process);
%plot_synergy_activation(learn_process.syn(4));

%% Comparison between raw and reconstructed EMG
plot_reconstruction(learn_process.syn(4));

%end



close all
clc
clear
arm = arm_model([45;90],0,0,0,0,0);
default_six_muscles(arm);

nn = nnetworkSRV(180,size(arm.R,2));

learn_process = learning_framework(nn,arm);
train_force_production(learn_process,200,[2]);
% train_force_annealing(learn_process,20);
plot_learned_force(learn_process,200);
fig_handles = plot_muscle_activations(learn_process,200);
plot_muscle_pulling_direction(learn_process,fig_handles);
plot_muscle_preferred_direction(learn_process,200,fig_handles);
plot_learning_error(learn_process,200);

identify_all_synergies(learn_process,200);

plot_rr_curve(learn_process);
plot_vaf_curve(learn_process);
% plot_vaf_muscle_curve(learn_process);
%
plot_synergy(learn_process.syn(4));
%
% plot_found_synergies(learn_process);
% plot_reconstruction(learn_process.syn(3));
%
% end
%
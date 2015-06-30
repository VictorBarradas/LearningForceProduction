close all
clc
clear
arm = arm_model([45;90],0,0,0,0,0);
default_six_muscles(arm);

nn = nnetworkSRV(180,size(arm.R,2));

%for j = 1:4
%nn = nnetwork(180,size(arm.R,2));
learn_process = learning_framework(nn,arm);
train_force_SRV(learn_process,10,[2]);
% %train_force_annealing(learn_process,20);

path = 'nnetworks';
files = dir([path, '/nn*.mat']);
num_model = length(files) + 1;
file_name = strcat('/nn',num2str(num_model));
save(strcat(path,file_name),'nn');


% plot_learned_force(learn_process,200,[2]);
% %plot_muscle_activations(learn_process,200,[4]);
% %plot_learning_error(learn_process,200,4);
% 
% 
% identify_all_synergies(learn_process,200,2);
% % 
% plot_rr_curve(learn_process);
% plot_vaf_curve(learn_process);
% % %plot_vaf_muscle_curve(learn_process);
% % 
% plot_synergy(learn_process.syn(3));
% % 
% % %plot_found_synergies(learn_process);
% plot_reconstruction(learn_process.syn(3));
% % 
% % %end
% 
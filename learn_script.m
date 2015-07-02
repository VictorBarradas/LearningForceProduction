
close all
clc
clear
arm = arm_model([45;90],0,0,0,0,0);
default_six_muscles(arm);

nn = nnetworkSRV(180,size(arm.R,2));

learn_process = learning_framework(nn,arm);
train_force_SRV(learn_process,200,[2]);

path = 'nnetworks';
files = dir([path, '/lp*.mat']);
num_model = length(files) + 1;
file_name = strcat('/lp',num2str(num_model));
save(strcat(path,file_name),'learn_process');

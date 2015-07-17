close all
clc
clear

arm = arm_model([45;90],0,0,0,0,0);
default_six_muscles(arm);

weight_levels = [1/32 1/8 1/2 1];

weights = npermutek(weight_levels,2);


for i = 1:size(weights,1) 
    costF(i) = cost_isometric_force(weights(i,:));
    nn(i) = nnetworkSRV(180,size(arm.R,2));
    learn_process(i) = learning_framework(nn(i),arm,costF(i));
    train_force_production(learn_process(i),200,2);
    plot_learned_force(learn_process(i),200)
end

save('test.mat','learn_process');

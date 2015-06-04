function [] = plot_force_direction(nInput,nTrainingGroup,W,armPosition,arm_physics)
%Function that plot the results of learning force production in all
%directions
%   Detailed explanation goes here

cInput = -180:360/nInput:180-360/nInput;
cTrainingGroup = -180:360/nTrainingGroup:180 - 360/nTrainingGroup;
sigmaNoise = 0.0*ones(nInput,1);
omegaInput = 5;

for i=1:nTrainingGroup
    desTheta = cTrainingGroup(i);
    angle(i) = desTheta;
    inputActivation = exp(-log(2)*(angle_subtraction(desTheta, cInput)/omegaInput).^2); %RBF
    inputNoise = max(0, normrnd(zeros(nInput,1), sigmaNoise));
    inputActivation = inputActivation + inputNoise;
    layerOutput = W'*inputActivation;
    muscleActivation = 1./(1 + exp(-1/10*(layerOutput)));
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

end


function [] = plot_force_direction(neural_network,arm_model,nPoints)
%Function that plot the results of learning force production in all
%directions
%   Detailed explanation goes here

nInput = neural_network.nInput;
cInput = neural_network.cInput;
cPoints = -180:360/nPoints:180 - 360/nPoints;
sigmaNoise = 0.0*ones(nInput,1);
omegaInput = neural_network.omegaInput;

for i=1:nPoints
    desTheta = cPoints(i);
    angle(i) = desTheta;
    inputActivation = exp(-log(2)*(angle_subtraction(desTheta, cInput)/omegaInput).^2); %RBF
    inputNoise = max(0, normrnd(zeros(nInput,1), sigmaNoise));
    inputActivation = inputActivation + inputNoise;
    layerOutput = neural_network.W'*inputActivation;
    muscleActivation = 1./(1 + exp(-1/10*(layerOutput)));
    producedForce = activation2force(arm_model,muscleActivation);
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

polar(cPoints*pi/180,producedMagnitude);
title('Learned force magnitude for every direction')
hold on

end


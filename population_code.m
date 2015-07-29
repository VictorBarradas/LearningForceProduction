function [activation] = population_code(nNeurons,direction,magnitude)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

cInput = -180:360/nNeurons:180-360/nNeurons;
omegaInput = 5;
sigmaInputUnits = 0.0*ones(nNeurons,1);


activation = exp(-log(2)*(angle_subtraction(direction,cInput)/omegaInput).^2);
% Noise injection
inputNoise = normrnd(zeros(nNeurons,1), sigmaInputUnits);
activation = max(0, activation + inputNoise);

end


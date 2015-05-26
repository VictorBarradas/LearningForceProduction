function [thetaResult] = angle_subtraction(theta1, theta2)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

n = length(theta2);
thetaResult = zeros(n,1);

for i = 1:n
    if theta1 - theta2(i) <= -180
        thetaResult(i) = -(- 180 - theta1) + (180 - theta2(i));
    elseif theta1 - theta2(i) >= 180
        thetaResult(i) = -((180 - theta1) - (- 180 - theta2(i)));
    else
        thetaResult(i) = theta1 - theta2(i);
    end
end

end


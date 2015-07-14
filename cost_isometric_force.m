function u = cost_isometric_force(weights,errorForce,muscleActivation)
u = weights(1)*errorForce + weights(2)*sum(muscleActivation.^2);
end


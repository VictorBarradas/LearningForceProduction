function [f] = arm_physics_four_muscles(q,a)
%Planar arm: 4 muscles, 2 actuating each joint
%   q - joint position vector (degrees)
%   a - activation vector

l_1 = 1;
l_2 = 1;

r = 1;

q = q*pi/180;
q_1 = q(1);
q_2 = q(2);

J = [-l_1*sin(q_1)-l_2*sin(q_1+q_2), -l_2*sin(q_1+q_2);
     l_1*cos(q_1)+l_2*cos(q_1+q_2), l_2*cos(q_1+q_2)];
 
JTI1 = transpose(inv(J));
JTI = inv(transpose(J));


R = [r -r 0 0;
     0 0 r -r];
 
F0 = [10 0 0 0;0 10 0 0;0 0 10 0;0 0 0 10];

muscle_force = F0*a;
torque = R*muscle_force; 

f = JTI1*torque;

tau_1 = torque(1);
tau_2 = torque(2);
% 
f(1) = (l_2*cos(q_1+q_2)*tau_1 - (l_1*cos(q_1) + l_2*cos(q_1+q_2))*tau_2)/(l_1*l_2*sin(q_2));
f(2) = (l_2*sin(q_1+q_2)*tau_1 - (l_1*sin(q_1) + l_2*sin(q_1+q_2))*tau_2)/(l_1*l_2*sin(q_2));

end


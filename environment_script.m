close all
clc
clear

state = [0;0;0.5;2];
parameters = [1;1;1];
ve = virtual_environment(state,parameters);

timestep = 0.05;
sim_time = 10;
current_time = 0;
h = 1;

while current_time <= sim_time
    forward_simulation(ve,[0.5;2],timestep);
    plot_cursor(ve,h);
    current_time = current_time + timestep;
    pause(timestep);
end
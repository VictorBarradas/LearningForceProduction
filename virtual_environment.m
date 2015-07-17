classdef virtual_environment < handle
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        % State
        position % Cursor position
        velocity % Cursor velocity
        % Dynamic parameters
        m % Particle mass
        k % Stiffness
        b % Damping factor
    end
    
    methods
        function obj = virtual_environment(state,parameters)
            initial_pos = state(1:2);
            initial_vel = state(3:4);
            obj.position = initial_pos;
            obj.velocity = initial_vel;
            obj.m = parameters(1);
            obj.k = parameters(2);
            obj.b = parameters(3);
        end
        
        function forward_simulation(obj,input,timestep)
            vel_x = obj.velocity(1);
            vel_y = obj.velocity(2);
            pos_x = obj.position(1);
            pos_y = obj.position(2);
            f_x = input(1);
            f_y = input(2);
            
            xdd = 1/obj.m*(f_x - obj.b*vel_x - obj.k*pos_x);
            xd = vel_x + timestep*xdd;
            x = pos_x + timestep*xd;
            
            ydd = 1/obj.m*(f_y - obj.b*vel_y - obj.k*pos_y);
            yd = vel_y + timestep*ydd;
            y = pos_y + timestep*yd;
            
            obj.velocity(1) = xd;
            obj.velocity(2) = yd;
            obj.position(1) = x;
            obj.position(2) = y;
        end
        
        function plot_cursor(obj,h)
            figure(h)
            plot(obj.position(1),obj.position(2),'.','MarkerSize',40);
            xlim([-3,3]);
            ylim([-3,3]);
            daspect([1 1 1])
        end
        
    end
    
end


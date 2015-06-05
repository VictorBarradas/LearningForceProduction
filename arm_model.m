classdef arm_model
    %Arm physics for static force production
    %   
    
    properties
        arm_position %Vector with joint angles (shoulder,elbow) (Degrees)
        l %Vector with arm segment length
        J %Jacobian of the arm
        R %Moment-arm matrix 
        F0 %Max force matrix
    end
    
    methods
        function obj = arm_model(arm_position,l,J,R,F0)
            obj.arm_position = arm_position*pi/180;
            obj.l = l;
            obj.J = J;
            obj.R = R;
            obj.F0 = F0;
        end
        
        function f = activation2force(obj,a)
            JIT = transpose(inv(obj.J));
            f = JIT*obj.R*obj.F0*a;
        end
        
        function default_four_muscles(obj)
            obj.l = [1;1];
            l_1 = obj.l(1);
            l_2 = obj.l(2);
            
            q_1 = obj.arm_position(1);
            q_2 = obj.arm_position(2);
            
            obj.J = [-l_1*sin(q_1)-l_2*sin(q_1+q_2), -l_2*sin(q_1+q_2);
                      l_1*cos(q_1)+l_2*cos(q_1+q_2), l_2*cos(q_1+q_2)];
            r = 1;
            obj.R = [r -r 0 0;
                     0 0 r -r];
            obj.F0 = diag([10;10;10;10]);
                 
        end
        
        function obj = default_six_muscles(obj)
            obj.l = [1;1];
            l_1 = obj.l(1);
            l_2 = obj.l(2);
            
            q_1 = obj.arm_position(1);
            q_2 = obj.arm_position(2);
            
            obj.J = [-l_1*sin(q_1)-l_2*sin(q_1+q_2), -l_2*sin(q_1+q_2);
                      l_1*cos(q_1)+l_2*cos(q_1+q_2), l_2*cos(q_1+q_2)];
            r = 1;
            obj.R = [r -r 0 0 r -r;
                     0 0 r -r r -r];
            obj.F0 = diag([10;10;10;10;10;10]);
        end
    end
    
end

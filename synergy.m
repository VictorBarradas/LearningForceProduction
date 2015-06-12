classdef synergy
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        rawEMG %Matrix containing raw EMG from trials
        W %Matrix with control coefficients for synergies
        H %Matrix with synergies
        var
    end
    
    methods
        function obj = synergy(remg)
            obj.rawEMG = remg;
        end
        
        function synergy_id(obj,nSynergies)
            %Scale emg channels for unit variance
            stdev = std(obj.rawEMG');
            emgScaled = diag(1./stdev)*obj.rawEMG;
            [Wpre,Hpre] = nnmf(emgScaled,nSynergies);
            %Rescaling
            WRescaled = diag(stdev)*Wpre;
            %Synergy vector normalization
            m=max(WRescaled);% vector with max activation values
            for i=1:nSynergies
                H(i,:)=Hpre(i,:)*m(i);
                W(:,i)=WRescaled(:,i)/m(i);
            end
        end
        
        function variance_within_synergies(obj)
            reconstruct = obj.W*obj.H;
            for i = 1:size(obj.H,1)
                obj.var(i) = 100*sum(sum(obj.W(:,i)*obj.H(i,:)))/sum(sum(reconstruct));
            end
        end
        
        function plot_synergy(obj)
            n = size(obj.H,1);
            figure;
            for i = 1:n
                subplot(n,1,i);
                bar(obj.H(i,:));
                legend(num2str(obj.var(i)));
            end
        end
        
    end
    
end


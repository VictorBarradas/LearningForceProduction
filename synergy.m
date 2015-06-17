classdef synergy < handle
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        rawEMG %Matrix containing raw EMG from trials
        W %Matrix with control coefficients for synergies
        H %Matrix with synergies
        var
        muscle_names %Names of muscles in the system
    end
    
    methods
        function obj = synergy(remg,musc_n)
            obj.rawEMG = remg;
            obj.muscle_names = musc_n;
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
                obj.H(i,:)=Hpre(i,:)*m(i);
                obj.W(:,i)=WRescaled(:,i)/m(i);
            end
        end
        
        function variance_within_synergies(obj)
            reconstruct = obj.W*obj.H;
            for i = 1:size(obj.H,1)
                obj.var(i) = 100*sum(sum(obj.W(:,i)*obj.H(i,:)))/sum(sum(reconstruct));
            end
        end
        
        function plot_synergy(obj)
            n = size(obj.W,2);
            figure;
            for i = 1:n
                subplot(n,1,i);
                bar(obj.W(:,i));
                ylim([0,1.2]);
                text(5.75,1.0,strcat(num2str(obj.var(i)),'%'));
                if i == 1
                    title(strcat(num2str(n),{' muscle synergies assumed'}));
                end
                if i == n
                    set(gca,'XTickLabel',obj.muscle_names);
                else
                    set(gca,'XTickLabel',{});
                end
            end
        end
        
        function plot_synergy_activation(obj)
            nPoints = size(obj.H,1);
            cPoints = -180:360/nPoints:180 - 360/nPoints;
            cPoints = cPoints';
            for i = 1:size(obj.W,1)
                figure
                polar(cPoints*pi/180,obj.H(:,i));
            end
        end
        
        function plot_reconstruction(obj)
            nPoints = size(obj.H,2);
            cPoints = -180:360/nPoints:180 - 360/nPoints;
            reconstruct = obj.W*obj.H;
            for i = 1:size(reconstruct,1)
                figure
                polar(cPoints*pi/180,ones(size(cPoints)),'k');
                hold on
                h1 = polar(cPoints*pi/180,obj.rawEMG(i,:));
                hold on
                h2 = polar(cPoints*pi/180,reconstruct(i,:),'r');
                legend([h1,h2],{'raw','reconstructed'},'Location','northeastoutside');
                title(obj.muscle_names(i));
            end
        end
        
    end
    
end


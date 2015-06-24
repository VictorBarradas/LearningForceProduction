classdef synergy < handle
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        rawEMG %Matrix containing raw EMG from trials
        nSynergies %Number of muscle synergies 
        W %Matrix with synergies
        H %Matrix with control coefficients for synergies
        recEMG %Reconstructed EMG data
        var
        muscle_names %Names of muscles in the system
    end
    
    methods
        function obj = synergy(remg,musc_n,n_syn)
            obj.rawEMG = remg;
            obj.muscle_names = musc_n;
            obj.nSynergies = n_syn;
        end
        
        function synergy_id(obj)
            %Scale emg channels for unit variance
            stdev = std(obj.rawEMG');
            emgScaled = diag(1./stdev)*obj.rawEMG;
            [Wpre,Hpre] = nnmf(emgScaled,obj.nSynergies);
            %Rescaling
            WRescaled = diag(stdev)*Wpre;
            %Synergy vector normalization
            m=max(WRescaled);% vector with max activation values
            for i=1:obj.nSynergies
                obj.H(i,:)=Hpre(i,:)*m(i);
                obj.W(:,i)=WRescaled(:,i)/m(i);
            end
            obj.recEMG = obj.W*obj.H;
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
        
        function rr = rsquared(obj)                        
            r = corr2(obj.rawEMG,obj.W*obj.H);
            rr = r^2;
        end
        
        function VAF = vaf_fit(obj)
            X=cat(3,obj.rawEMG,obj.W*obj.H);    
            UR=(sum(sum(prod(X,3))))^2/(sum(sum((obj.rawEMG).^2))*sum(sum((obj.W*obj.H).^2)));
            VAF=100*UR;
        end
        
        function VAF = vaf_fit_muscles(obj)
            for i = 1:size(obj.rawEMG,1)
                X = [obj.rawEMG(i,:) obj.recEMG(i,:)];
                VAF(i) = sum((prod(X,1)))^2 / (sum(obj.rawEMG(i,:).^2)*sum(obj.recEMG(i,:).^2)); %regression sum of squares/total sum of squares
            end
            VAF = 100*VAF;
        end
        
    end
    
end


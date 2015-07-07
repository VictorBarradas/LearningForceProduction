function [] = plot_synergy_comparison(syn,r)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

figure

nSyn = size(syn{1},2);
nMusc = size(syn{1},1);
nComp = length(syn);
color = jet(nSyn);

for i = 1:nComp
    for j = 1:nSyn
        
        subplot(nSyn,nComp,i + (j-1)*nComp);
        bar(syn{i}(:,j),'FaceColor',color(j,:));
        ylim([0,1.2]);
        xlim ([0,nMusc+1]);       
        %         text(5.75,1.0,strcat(num2str(obj.var(j)),'%'));
        
        if i ~= 1
            title(strcat({'r = '},num2str(r(j,i))));
            set(gca,'YTickLabel',{});  
        end
        if j ~= nSyn
            set(gca,'XTickLabel',{});  
        end
    end
    
end
end


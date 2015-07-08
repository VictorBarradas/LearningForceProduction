%% Synergy comparison
% Comparison of synergies obtained by different re-runs of the nnmf 
% algorithm on the same data set (3 synergies)
close all
clc
clear

load('nnetworks/lp1.mat')

nSynMat = 100; % Number of synergy matrices
syn = cell(nSynMat,1); % Cell containing synergy matrices
nSelectedSyn = 3;

i = 1;

% Run nnmf algorithm nSynMat times to get different solutions
while i <= nSynMat
    identify_individual_synergy(learn_process,200,nSelectedSyn,2);   
    syn{i} = learn_process.syn(nSelectedSyn).W;
    nNan = sum(sum(isnan(syn{i})));
    if nNan == 0
        i = i+1;
    end 
end

% Order individual synergies within the synergy matrix
for i = 1:nSynMat
    [~,pos] = compare_synergies(syn{1},syn{i});
    syn{i} = syn{i}(:,pos(:,2)');
end

% Compare the solutions
r = zeros(size(syn{1},2),1);
meanR = 0;
for i = 1:nSynMat
    r(:,i) = compare_synergies(syn{1},syn{i});
end

meanR = mean(r,2);

nPlots = 6;
random_syn = [1; randi(nSynMat-1,[nPlots - 1,1]) + 1];

syn_plot = syn(random_syn);
r_plot = r(:,random_syn);

plot_synergy_comparison(syn_plot,r_plot);

%% Average similarity of 100 re-runs to the first obtained synergy

meanR



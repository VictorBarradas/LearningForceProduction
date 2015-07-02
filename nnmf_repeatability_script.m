close all
clc
clear

load('nnetworks/lp1.mat')

nSynMat = 100; % Number of synergy matrices
syn = cell(nSynMat,1); % Cell containing synergy matrices

i = 1;

% Run nnmf algorithm nSynMat times to get different solutions
while i <= nSynMat
    identify_individual_synergy(learn_process,200,4,2);   
    syn{i} = learn_process.syn(4).W;
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
for i = 1:nSynMat
    r = r + compare_synergies(syn{1},syn{i});
end

r = r/nSynMat;



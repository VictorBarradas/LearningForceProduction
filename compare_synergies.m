function [r,pos] = compare_synergies(syn1,syn2)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

for i = 1:size(syn1,2);
    syn1_norm(:,i) = syn1(:,i)/norm(syn1(:,i));
    syn2_norm(:,i) = syn2(:,i)/norm(syn2(:,i));
end

w = syn1_norm'*syn2_norm;

[r,pos] = max(w,[],2);

enum = 1:1:size(syn1,2);
enum = enum';
pos = [enum pos];

end


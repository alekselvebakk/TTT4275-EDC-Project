%% Part 2, Task 2
clc; clear all; close all;

% Initialization
if ~exist('num_test','var')
    load("MNist_ttt4275/data_all.mat");
end

M = 64;  % number of clusters per class 
N = 784; % number of features per vector
I = 10;  % number of classes

%% Task 2a: Clustering training vectors

idx = cell(10,1);
C   = cell(10,1);

tic
for i = 1:I
    [idx{i}, C{i}] = kmeans(trainv((trainlab+1) == i,:),M);
end
T_cluster = toc;

%% Task 2b: Running NN classification using clusters

% this one looks at only the best fit for each number
% then chooses the best fit from each number
% should be subject to outliers where a single cluster has an exceptional
% fit for one class, but the average fit is much better for another class

tic
best_dist_class    = zeros(I, num_test);
results_dist_      = zeros(1,num_test);
results_class_offs = zeros(1,num_test);

for i = 1:I
    best_dist_class(i,:) = min(dist(C{i}, testv'));
end

[results_dist_, results_class_offs] = min(best_dist_class);
results = results_class_offs' - 1;

NN_cluster_confmat  = confusionmat(testlab, results);
NN_cluster_err_rate = (num_test - trace(NN_cluster_confmat))/num_test;
T_NN_cluster        = toc;

%% Task 2c: Running KNN, K=7

% This one looks at the 7 best fits from all kmeans clusters
% then does a majority vote among these. I'm not sure what to call it, 
% but this one definitely does too much smoothing.
% Kmeans should already have smoothed out outliers from training set.
% thus, doing a majority vote on these smoothed templates means that you
% end up with a "twice smoothed" decision.

% Peformance here is slightly worse than the NN_cluster variety used in
% task 2b: roughly 1% more error, and is usually half a second faster. 

% This method probably useful for cases where the variance 
% within each class is much bigger and the kmeans clusters
% are liable to overlap between classes.

% Might also be more useful as the number of clusters per class
% increases as this increases the likelihood of outliers that 
% coincidentally have the shortest distance. 

tic
K = 7;

all_distances = zeros(I*M,num_test);

for i = 1:I
    all_distances(1 + M*(i-1):M*i,:) = dist(C{i}, testv');
end

[B, I_offs] = mink(all_distances, 7);
Votes = idivide(uint16(I_offs - 1),64);
[Modes, Freqs] = mode(Votes);

% ---------- RESOLVING MODE CONFLICTS -----------
% Look for conflicts where mode has 3 or less votes
for i = find(Freqs <= 3)
    % count occurrences of all votes 
    [votecounts, edges_] = histcounts(Votes(:,i), -0.5:1:9.5);
    % if the majority vote is contested
    if size(find(votecounts == Freqs(i)),2)
        %find closest neighbour which is also a candidate
        for j = 1:K
            best_candidate = Votes(j,i);
            if votecounts(best_candidate + 1) == Freqs(i)
                break;
            end    
        end
        Modes(1,i) = best_candidate;
    end
end

results = Modes';

KNN_confmat   = confusionmat(uint16(testlab), results);
KNN_err_rate  = (num_test - trace(KNN_confmat))/num_test;
T_KNN_cluster = toc;

%% Save results

save("all_results_part2_task2.mat", "T_cluster", ...
    "T_NN_cluster", "NN_cluster_confmat", "NN_cluster_err_rate", ...
    "T_KNN_cluster", "KNN_confmat",  "KNN_err_rate");

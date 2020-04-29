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

for i = 1:I
    [idx{i}, C{i}] = kmeans(trainv((trainlab+1) == i,:),M);
end

%% Task 2b: Running NN classification using clusters

% this one looks at only the best fit for each number
% then chooses the best fit from each number
% should be subject to outliers where a single cluster has an exceptional
% fit for one number, but the average fit is much better for another class

tic
best_dist_class      = zeros(I, num_test);
results_dist         = zeros(1,num_test);
results_class_offs   = zeros(1,num_test);

for i = 1:I
    best_dist_class(i,:) = min(dist(C{i}, testv'));
end

[results_dist, results_class_offs] = min(best_dist_class);
results = results_class_offs' - 1;

confmat  = confusionmat(testlab, results)
err_rate = (num_test - trace(confmat))/num_test
toc

%% Task 2c: Running KNN, K=7
% similar to above task, do majority vote among 7 best of all distances

% this one looks at the 7 best fits from kmeans overall
% then does a majority vote among these
% not sure what to call it, but this one definitely does too much fitting
% kmeans should already have smoothed out outliers from training set.
% thus, doing a majority vote on these smoothed templates means that you
% end up with a "twice smoothed" decision.

% this is probably useful for cases where the variance within the classes
% is much bigger and the kmeans templates are liable to overlap between
% classes

% might also be more useful as the number of clusters per class increases
% as this increases the likehood of outliers that coincidentally have the 

tic
K = 7;

all_distances = zeros(I*M,num_test);

for i = 1:I
    all_distances(1 + M*(i-1):M*i,:) = dist(C{i}, testv');
end

[B_, I_offs] = mink(all_distances, 7);
results = mode(idivide(uint16(I_offs - 1),64))';

KNN_confmat  = confusionmat(uint16(testlab), results)
KNN_err_rate = (num_test - trace(KNN_confmat))/num_test
toc

%% Task2c: again with proper dispute handling
tic
K = 7;

all_distances = zeros(I*M,num_test);

for i = 1:I
    all_distances(1 + M*(i-1):M*i,:) = dist(C{i}, testv');
end

[B, I_offs] = mink(all_distances, 7);
Votes = idivide(uint16(I_offs - 1),64);
[Modes, Freqs] = mode(Votes);

 % Look for conflicts where mode has freq <= 3
for i = find(Freqs <= 3)
    
    candidates = zeros(1,1);
    votes = Votes(:,i);
    j = 1;
    
    % fill candidates vector with all classes 
    % that have the same freq as inital mode
    while 1
        [m, f] = mode(votes);
        if f ~= Freqs(i)
            break;
        end
        candidates(j) = m;
        for k = find(votes == m)
            votes(k) = 10 + k;
        end
        j = j + 1;
    end
    
    % for only one candidate, go to next potential conflict
    if size(candidates, 2) == 1
        continue;
    end
    
    % find closest neighbour which is also a candidate
    for j = 1:K
        best_candidate = idivide(uint16(I_offs(j,i) - 1),64);
        if nnz(candidates == best_candidate) > 0 
            break;
        end    
    end
    
    % set decision as closest neighbour which is also a candidate
    Modes(1,i) = best_candidate;
end

results = Modes';
KNN_2_confmat  = confusionmat(uint16(testlab), results)
KNN_2_err_rate = (num_test - trace(KNN_2_confmat))/num_test
toc
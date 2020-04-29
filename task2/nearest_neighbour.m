%% Part 2, Task 1
clc; clear all; close all;

% Initialization
if ~exist('num_test','var')
    load("MNist_ttt4275/data_all.mat");
end

N            = 784;  % number of features per vector
I            = 10;   % number of classes
chunk_size   = 1000; % number of images per chunk
train_chunks = num_train/chunk_size;
test_chunks  = num_test/chunk_size;

% Note that chunk size should be set S.T. num_train and num_test
% are both divisible by chunk size, as the program 
% DOES NOT validate automatic index calculation.

 %% Task 1a: classify and find confusion matrix without majority voting

tic

% Helper function for chunk index ranges
i_rng = @(i) 1 + chunk_size*(i-1):chunk_size*i;

% Preallocation
results_list        = zeros(num_test,1,'uint8');
inner_chunk_scores  = zeros(train_chunks, chunk_size);
inner_chunk_indices = zeros(train_chunks, chunk_size);
indices             = zeros(1,chunk_size);

for i = 1:test_chunks
    
    fprintf("Testing subset: " + num2str(i) + "\n");
    
    for j = 1:train_chunks
        
        fprintf("\tTraining subset: " + num2str(j) + "\n");
        
        [inner_chunk_scores(j,:), inner_chunk_indices(j,:)] = ...
            min(dist(trainv(i_rng(j),:), testv(i_rng(i),:)'));
        
    end
    
    [~, chunk_indices] = min(inner_chunk_scores);
    
    % didn't find a more clever way to do this assignment sadly
    for j = 1:chunk_size
        indices(j) = ...
            (chunk_indices(j)-1)*chunk_size + ...
            inner_chunk_indices(chunk_indices(j),j);
    end
    
    results_list(i_rng(i)) = trainlab(indices);
end

% find confusion matrix and error rate
NN_conf_mat = confusionmat(uint8(testlab), results_list);
NN_err_rate = 1 - trace(NN_conf_mat)/num_test;
T_NN        = toc;

% save results
save("all_results_part2_task1_TEST.mat",...
     "T_NN", "NN_conf_mat",  "NN_err_rate");

%% Task 1b & 1c: plotting correct and incorrect classifications

% Finding images to plot
error_indices  = zeros(10,1,"uint16");
valid_indices  = zeros(10,1,"uint16");

for i=1:num_test
    actual = testlab(i) + 1;      % offset by 1 for indexing
    est    = results_list(i) + 1; % offset by 1 for indexing
 
    if valid_indices(actual) == 0 && actual == est
        valid_indices(actual) = i;
    elseif error_indices(actual) == 0 && actual ~= est
        error_indices(actual) = i;
    end
    
    if nnz([error_indices; valid_indices]) == 2*I
       break; 
    end
end

% plotting correct
figure(1);
for i = 1:10
    if i == 1
        index = 2;
    else
        index = i + 2;
    end
    subplot(4,3,index);
    
    x = zeros(28, 28, "uint8");
    x(:) = testv(valid_indices(i),:);
    image(x');
    title("Answer: " + num2str(i-1));
end
sgtitle("Correctly classified numbers, NN both chunks 1k");

% plotting incorrect
figure(2);
for i = 1:10
    if i == 1
        index = 2;
    else
        index = i + 2;
    end
    subplot(4,3,index);
    x = zeros(28, 28, "uint8");
    x(:) = testv(error_indices(i),:);
    image(x');
    title("Answer: " + num2str(i-1) + ...
          ", classified as: " + num2str(results_list(error_indices(i))));
end
sgtitle("Incorrectly classified numbers, NN both chunks 1k");
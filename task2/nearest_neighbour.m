
%% initialization

load("./MNist_ttt4275/data_all.mat")
subset_train = cell(60,1);
subset_test = cell(10,1);

for i = 1:60
    if i < 11
        subset_test{i} = testv(1 + (i-1)*1000:(i*1000),:)';
    end
    subset_train{i} = trainv(1 + (i-1)*1000:(i*1000),:);
end

% counting containers:
% start with 1000 by 1000 matrix
% for each of the test subsets, 60 of these matrices are produced
% vertical min_index of this matrix will give the image number in that
% subset which matches best. 

results_list = zeros(10000,1,'int8');

%% classifier

% solver:
% collect votes from each subset, then choose most popular number
for i = 1:10
    fprintf("testing subset: " + num2str(i) + "\n");
    temp_count_mat = zeros(1000, 10, 'int8');
    for j = 1:60
        fprintf("\ttraining subset: " + num2str(j) + "\n");
        A = dist(subset_train{j}, subset_test{i});
        [M, subs_I] = min(A);
        I = int32(subs_I' + (j-1)*1000);
        values = int8(trainlab(I));
        for k = 1:1000
            temp_count_mat(k,values(k)+1) = ...
                temp_count_mat(k,values(k)+1) + 1;
        end
    end
    % note that temp_count_mats indices are offset by 1..
    [M, num_offs_I] = max(temp_count_mat, [], 2);
    num_I = int8(num_offs_I - 1);
    results_list(1 + (i-1)*1000: i*1000) = num_I;
end

save("NN_allSamples_both1kChunks.mat","results_list");

%% find confusion matrix and error rate
% confusion matrix is, real values on vertical axis, probabilities 
% for that given value on horizontal axis.

%load("NN_allSamples_both1kChunks.mat")
%load("./MNist_ttt4275/data_all.mat")

matrix_counter = zeros(10,10,"int16");
total_counter  = zeros(10,1,"int16");
N              = size(testlab,1);

for i=1:N
    actual = testlab(i) + 1;      % offset by 1 for indexing
    est    = results_list(i) + 1; % offset by 1 for indexing
    total_counter(actual) = total_counter(actual) + 1;
    matrix_counter(actual,est) = ...
        matrix_counter(actual,est) + 1;
end

confusion_matrix = matrix_counter;
error_rate = 1 - trace(confusion_matrix)/N;
save("NN_confusion_and_errRate_allSamples_both1kChunks.mat",...
    "confusion_matrix","error_rate");

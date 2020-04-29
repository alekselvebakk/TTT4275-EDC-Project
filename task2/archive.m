%% Part2, Task 1a: classify wrong implementation

% classifier
tic

% Helper function for chunk index ranges
i_rng = @(i) 1 + chunk_size*(i-1):chunk_size*i;

% Preallocation
results_list = zeros(10000,1,'uint8');
all_votes = zeros(train_chunks, chunk_size, "uint8");

for i = 1:test_chunks
    
    fprintf("Testing subset: " + num2str(i) + "\n");
    
    for j = 1:train_chunks
        
        fprintf("\tTraining subset: " + num2str(j) + "\n");
        
        [~, I_offs] = min(dist(trainv(i_rng(j),:), testv(i_rng(i),:)'));
        votes = uint8(trainlab(I_offs' + (j-1)*chunk_size));
        all_votes(j,:) = votes';
        
    end
    
    [M, F] = mode(all_votes);
    for k = find(F <= train_chunks/2)
        fprintf("Index %d has a potential conflict" + ...
                ", mode has %d votes.\n", k + (i-1)*chunk_size, F(k));
    end
    results_list(i_rng(i)) = M';
end

% find confusion matrix and error rate
NN_conf_mat = confusionmat(uint8(testlab), results_list);
NN_err_rate = 1 - trace(NN_conf_mat)/num_test;
T_NN        = toc;

% save results
save("all_results_part2_task1.mat",...
     "T_NN", "NN_conf_mat",  "NN_err_rate");
 
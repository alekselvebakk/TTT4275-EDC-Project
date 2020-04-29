clc; clear all; close all;

load("all_results_part2_task1.mat");
load("all_results_part2_task2.mat");

fprintf("NN using all training vectors as templates:\n");
fprintf("\tTime spent: %d minutes.\n",idivide(uint16(T_NN),60));
fprintf("\tError rate: %1.4f \n",NN_err_rate);
fprintf("\tConfusion matrix:\n");
disp(NN_conf_mat);

fprintf("NN using 64 clusters per class as templates:\n");
fprintf("\tTime spent clustering:  %2.2f seconds.\n",   T_cluster);
fprintf("\tTime spent classifying: %2.2f seconds.\n",T_NN_cluster);
fprintf("\tError rate:             %1.4f \n", NN_cluster_err_rate);
fprintf("\tConfusion matrix:\n");
disp(NN_cluster_confmat);

fprintf("KNN using 64 clusters per class as templates:\n");
fprintf("\tTime spent clustering:  %2.2f seconds.\n",    T_cluster);
fprintf("\tTime spent classifying: %2.2f seconds.\n",T_KNN_cluster);
fprintf("\tError rate:             %1.4f \n",         KNN_err_rate);
fprintf("\tConfusion matrix:\n");
disp(KNN_confmat);
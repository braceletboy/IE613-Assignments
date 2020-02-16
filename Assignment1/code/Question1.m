%% Full information setting
%   This script contains the main code for simulating the full information 
%   setting in Question 1.
%
%% Simulation of the Weighted Majority Algorithm
%
close all;
num_experts = 10;
horizon = 10^5;
num_sample_paths = 50;
delta = 0.1;
eta_values = (0.1:0.2:2.1)*sqrt(2*log(num_experts)/horizon);
regret_vectors = zeros(length(eta_values), num_sample_paths);
parfor eta_idx = 1:length(eta_values)
    eta = eta_values(eta_idx);
    loss_incurred_algorithm = zeros(num_sample_paths, 1);
    loss_incurred_expertwise = zeros(num_sample_paths, num_experts);
    for path_idx = 1:num_sample_paths
        % initialize the algorithm
        weights = ones(1, num_experts)/num_experts;
        % simulate
        for round = 1:horizon
            arm_played = randsample(10, 1, true, weights);
            losses = get_full_loss(round, horizon, delta);
            loss_incurred_expertwise(path_idx, :) = ...
                 loss_incurred_expertwise(path_idx, :) + losses;
            loss_incurred_algorithm(path_idx) = ...
                loss_incurred_algorithm(path_idx) + dot(weights, losses);
            weights = weights.*exp(-1*eta*losses);
            weights = weights/sum(weights, 'all');
        end
    end
    regret_vectors(eta_idx,:) = loss_incurred_algorithm - ...
        min(loss_incurred_expertwise, [], 2);
end
figure;
title("Full Information Setting - Weighted Majority Algorithm");
plot_with_errorbar(regret_vectors, eta_values, "WMA");
xlabel("eta");
ylabel("regret");
legend;
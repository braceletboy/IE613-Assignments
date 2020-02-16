%% Bandit information setting
%   This script contains the main code for simulating the bandit info 
%   setting in Question 2.
%
%% Simulation of the EXP3, EXP3P and EXP3IX Algorithms
%
close all;
num_arms = 10;
horizon = 10^5;
num_sample_paths = 50;
delta = 0.1;
eta_values = (0.1:0.2:2.1)*sqrt(2*log(num_arms)/num_arms/horizon);
regret_vectors_exp3 = zeros(length(eta_values), num_sample_paths);
regret_vectors_exp3P = zeros(length(eta_values), num_sample_paths);
regret_vectors_exp3IX = zeros(length(eta_values), num_sample_paths);
parfor eta_idx = 1:length(eta_values)
    eta = eta_values(eta_idx);
    beta_exp3P = eta;
    gamma_exp3P = num_arms*eta;
    gamma_exp3IX = eta/2;
    
    estimated_losses_exp3 = zeros(num_sample_paths, num_arms);
    estimated_losses_exp3P = zeros(num_sample_paths, num_arms);
    estimated_losses_exp3IX = zeros(num_sample_paths, num_arms);
    
    losses_incurred_exp3 = zeros(num_sample_paths, 1);
    losses_incurred_exp3P = zeros(num_sample_paths, 1);
    losses_incurred_exp3IX = zeros(num_sample_paths, 1);
    
    % simulate
    for path_idx = 1:num_sample_paths
        for round = 1:horizon
            % calculate the probabilities for EXP3
            weights_exp3 = exp(-1*eta*estimated_losses_exp3(path_idx,:));
            weights_exp3 = weights_exp3/sum(weights_exp3, 'all');
            
            % calculate the probabilities for EXP3P
            weights_exp3P = exp(-1*eta*estimated_losses_exp3P(path_idx,:));
            weights_exp3P = (1-gamma_exp3P)*weights_exp3P/...
                sum(weights_exp3P) + gamma_exp3P/num_arms;
            
            % calculate the probabiliteis for EXP3IX
            weights_exp3IX = exp(-1*eta*estimated_losses_exp3IX(...
                path_idx,:));
            weights_exp3IX = weights_exp3IX/sum(weights_exp3IX, 'all');
            
            % play arm according to probabilities
            arm_played_exp3 = randsample(10, 1, true, weights_exp3);
            arm_played_exp3P = randsample(10, 1, true, weights_exp3P);
            arm_played_exp3IX = randsample(10, 1, true, weights_exp3IX);
            
            % get loss
            loss_exp3 = get_bandit_loss(arm_played_exp3, round, horizon,...
                delta);
            loss_exp3P = get_bandit_loss(arm_played_exp3P, round, horizon,...
                delta);
            loss_exp3IX = get_bandit_loss(arm_played_exp3IX, round, horizon,...
                delta);
            
            % update step
            estimated_losses_exp3(path_idx, arm_played_exp3) = ...
                estimated_losses_exp3(path_idx, arm_played_exp3) + ...
                    loss_exp3/weights_exp3(arm_played_exp3);
            estimated_losses_exp3P(path_idx, arm_played_exp3P) = ...
                estimated_losses_exp3P(path_idx, arm_played_exp3P) + ...
                    loss_exp3P/weights_exp3P(arm_played_exp3P);
            estimated_losses_exp3P(path_idx, :) = ...
                estimated_losses_exp3P(path_idx, :) + ...
                    beta_exp3P./weights_exp3P;
            estimated_losses_exp3IX(path_idx, arm_played_exp3IX) = ...
                estimated_losses_exp3IX(path_idx, arm_played_exp3IX) + ...
                    loss_exp3IX/(weights_exp3IX(arm_played_exp3IX)+ ...
                        gamma_exp3IX);
            
            % calculate the cummulative loss incurred
            losses_incurred_exp3(path_idx) = ...
                losses_incurred_exp3(path_idx) + loss_exp3;
            losses_incurred_exp3P(path_idx) = ...
                losses_incurred_exp3P(path_idx) + loss_exp3P;
            losses_incurred_exp3IX(path_idx) = ...
                losses_incurred_exp3IX(path_idx) + loss_exp3IX;
        end
    end
    regret_vectors_exp3(eta_idx,:) = losses_incurred_exp3 - ...
        min(estimated_losses_exp3, [], 2);
    regret_vectors_exp3P(eta_idx,:) = losses_incurred_exp3P - ...
        min(estimated_losses_exp3P, [], 2);
    regret_vectors_exp3IX(eta_idx,:) = losses_incurred_exp3IX - ...
        min(estimated_losses_exp3IX, [], 2);
end
figure;
hold on;
title("Bandit Information Setting - EXP3 vs EXP3P vs EXP3IX");
plot_with_errorbar(regret_vectors_exp3, eta_values, "EXP3");
plot_with_errorbar(regret_vectors_exp3P, eta_values, "EXP3P");
plot_with_errorbar(regret_vectors_exp3IX, eta_values, "EXP3IX");
xlabel("eta");
ylabel("regret");
legend;
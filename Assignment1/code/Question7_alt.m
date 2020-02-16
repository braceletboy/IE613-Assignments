%% Two-armed Stochastic Bandit setting
%
%% Testing 
%
horizon = 10^5;
num_arms = 2;
num_sample_paths = 50;
eta_values = (0.1:0.2:2.1)*sqrt(2*log(num_arms)/num_arms/horizon);
regret_vectors_stochastic = zeros(length(eta_values), num_sample_paths);
regret_vectors_predefined = zeros(length(eta_values), num_sample_paths);
parfor eta_idx = 1:length(eta_values)
    eta = eta_values(eta_idx);
    
    estimated_losses_stochastic = zeros(num_sample_paths, num_arms);
    estimated_losses_predefined = zeros(num_sample_paths, num_arms);
    
    losses_incurred_stochastic = zeros(num_sample_paths, 1);
    losses_incurred_predefined = zeros(num_sample_paths, 1);
    
    % simulate
    for path_idx = 1:num_sample_paths
        for round = 1:horizon
            % calculate the probabilities
            weights_stochastic = exp(-1*eta*...
                estimated_losses_stochastic(path_idx,:));
            weights_predefined = exp(-1*eta*...
                estimated_losses_predefined(path_idx,:));
            
            % normalize the probabilities
            weights_stochastic = weights_stochastic/sum(...
                weights_stochastic, 'all');
            weights_predefined = weights_predefined/sum(...
                weights_predefined, 'all');
            
            % play arm according to probabilities
            arm_played_stochastic = randsample(2, 1, true, ...
                weights_stochastic);
            arm_played_predefined = randsample(2, 1, true, ...
                weights_predefined);
            
            % get loss
            loss_stochastic = get_loss(arm_played_stochastic);
            loss_predefined = get_predefined_loss(arm_played_predefined,...
                round);
            
            % update step
            estimated_losses_stochastic(path_idx, arm_played_stochastic) = ...
                estimated_losses_stochastic(path_idx, arm_played_stochastic) + ...
                    loss_stochastic/weights_stochastic(arm_played_stochastic);
            estimated_losses_predefined(path_idx, arm_played_predefined) = ...
                estimated_losses_predefined(path_idx, arm_played_predefined) + ...
                    loss_predefined/weights_predefined(arm_played_predefined);
            
            % calculate the cummulative loss incurred
            losses_incurred_stochastic(path_idx) = ...
                losses_incurred_stochastic(path_idx) + loss_stochastic;
            losses_incurred_predefined(path_idx) = ...
                losses_incurred_predefined(path_idx) + loss_predefined;
        end
    end
    regret_vectors_stochastic(eta_idx,:) = losses_incurred_stochastic - ...
        min(estimated_losses_stochastic, [], 2);
    regret_vectors_predefined(eta_idx,:) = losses_incurred_predefined - ...
        min(estimated_losses_predefined, [], 2);
end
figure;
hold on;
title("Two armed stochastic bandit");
plot_with_errorbar(regret_vectors_stochastic, eta_values, "stochastic");
plot_with_errorbar(regret_vectors_predefined, eta_values, "predefined");
xlabel("eta");
ylabel("regret");
legend;

%%-----------------------------------------------------------------------%%

function loss = get_loss(arm)
%% Return the bernoulli loss given the arm pulled. 
%
if arm == 1
    loss = binornd(1,0.5);
elseif arm == 2
    loss = binornd(1,0.55);
else
    error("arm can either be 1 or 2");
end
end

function loss = get_predefined_loss(arm, round)
%% Return predefined sequence of losses based on the arm pulled.
%
if arm == 1
    if round<=10^5/4
        loss = 1;
    else
        loss = 0;
    end
elseif arm == 2
    if round>10^5/4
        loss = 1;
    else
        loss = 0;
    end
end
end
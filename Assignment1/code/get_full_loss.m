function losses = get_full_loss(curr_round, horizon, delta)
%% Return the loss vector for the full information setting.
%
% SYNTAX:
%   rewards = get_full_reward(curr_round, horizon, num_experts, delta)i
%
% INPUTS:
%   curr_round = The current round number.
%   horizon = The total number of rounds in the bandit game.
%   num_experts = The number of experts in the game.
%   delta = The delta specified in the assignment.
%% Code
%
num_experts = 10;
losses = zeros(1, num_experts);
losses(1:8) = binornd(1, ones(1,8)*0.5);
losses(9) = binornd(1, 0.5-delta);
if curr_round<=horizon/2
    losses(10) = binornd(1, 0.5+delta);
else
    losses(10) = binornd(1, 0.5-2*delta);
end
end
function loss = get_bandit_loss(arm, curr_round, horizon, delta)
%% Return reward from a bernoulli distribution for given arm
%
% SYNTAX:
%   reward = get_bandit_reward(mean)
%
% INPUT:
%   arm = The arm for which we need the reward.
%   curr_round = The current round we are in.
%   horizon = The horizon for the bernoulli experiment.
%   delta = The delta value specified in the assignment.
%
% OUTPUT:
%   reward - The reward generated.
%% Code
%
if arm==10
    if curr_round <= horizon/2
        loss = binornd(1, 0.5+delta);
    else
        loss = binornd(1, 0.5-2*delta);
    end
elseif arm==9
    loss = binornd(1, 0.5-delta);
else
    loss = binornd(1, 0.5);
end
end
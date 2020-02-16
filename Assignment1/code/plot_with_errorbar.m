function plot_with_errorbar(regret_vectors, x_values, display_name)
%% Plots the regret with 95% confidence error bars against x_values
%
% SYNTAX:
%   plot_with_errorbar(regret_vectors, x_values)
%
% INPUTS:
%   regret_vectors = Matrix containing rows of regret vectors - each
%   row corresponds to a certain x_value.
%   x_values = Vector of x_values for the plot.
%
%
%% Plotting code
%
num_samples = size(regret_vectors, 2);
mean_regret_values = mean(regret_vectors, 2);
error_values = 1.960/sqrt(num_samples)*std(regret_vectors, 0, 2);
errorbar(x_values, mean_regret_values, error_values, '.-', ...
    'DisplayName', display_name);
end
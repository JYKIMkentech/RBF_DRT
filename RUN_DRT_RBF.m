clc; clear; close all;

%% Description
% This script performs DRT estimation with uncertainty analysis using bootstrap (RBF-based).
% It loads the data, allows user selection, and then applies RBF-based DRT estimation.
% Finally, it plots the estimated gamma with uncertainty bounds.

%% Graphic Parameters
axisFontSize = 14;
titleFontSize = 12;
legendFontSize = 12;
labelFontSize = 12;

%% Load Data
file_path = 'G:\공유 드라이브\Battery Software Lab\Projects\DRT\SD_lambda\';
mat_files = dir(fullfile(file_path, '*.mat'));

% Load all .mat files
for file = mat_files'
    load(fullfile(file_path, file.name));
end

%% Parameters
AS_structs = {AS1_1per_new, AS1_2per_new, AS2_1per_new, AS2_2per_new};
AS_names = {'AS1_1per_new', 'AS1_2per_new', 'AS2_1per_new', 'AS2_2per_new'};
Gamma_structs = {Gamma_unimodal, Gamma_unimodal, Gamma_bimodal, Gamma_bimodal};

fprintf('Available datasets:\n');
for idx = 1:length(AS_names)
    fprintf('%d: %s\n', idx, AS_names{idx});
end
dataset_idx = input('Select a dataset to process (enter the number): ');

AS_data = AS_structs{dataset_idx};
AS_name = AS_names{dataset_idx};
Gamma_data = Gamma_structs{dataset_idx};

types = unique({AS_data.type});
disp('Select a type:');
for i = 1:length(types)
    fprintf('%d. %s\n', i, types{i});
end
type_idx = input('Enter the type number: ');
selected_type = types{type_idx};

type_indices = find(strcmp({AS_data.type}, selected_type));
type_data = AS_data(type_indices);

num_scenarios = length(type_data);
SN_list = [type_data.SN];

fprintf('Selected dataset: %s\n', AS_name);
fprintf('Selected type: %s\n', selected_type);
fprintf('Scenario numbers: ');
disp(SN_list);

c_mat = lines(num_scenarios);

OCV = 0;
R0 = 0.1;

%% RBF Parameters
rbf_type = 'gaussian';   % Choose RBF type
shape_param = 1.0;       % Shape parameter for RBF

%% DRT and Uncertainty Estimation
gamma_discrete_true = Gamma_data.gamma;  % keep as column vector
theta_true = Gamma_data.theta;           % keep as column vector

gamma_est_all = cell(num_scenarios, 1);
V_est_all = cell(num_scenarios, 1);
V_sd_all = cell(num_scenarios, 1);
theta_discrete_all = cell(num_scenarios, 1);
gamma_lower_all = cell(num_scenarios, 1);
gamma_upper_all = cell(num_scenarios, 1);

num_resamples = 1000;

for s = 1:num_scenarios
    fprintf('Processing %s Type %s Scenario %d/%d...\n', AS_name, selected_type, s, num_scenarios);

    scenario_data = type_data(s);
    V_sd = scenario_data.V(:);
    ik = scenario_data.I(:);
    t = scenario_data.t(:);
    dt = scenario_data.dt;
    dur = scenario_data.dur;
    n = scenario_data.n;
    lambda = 30; % scenario_data.Lambda_hat; % Modify if necessary

    % RBF-based DRT estimation
    [gamma_est, V_est, theta_discrete, tau_discrete, ~, ~] = DRT_estimation_RBF(t, ik, V_sd, lambda, n, dt, dur, OCV, R0, rbf_type, shape_param);

    % Store results as column vectors (no transpose)
    gamma_est_all{s} = gamma_est;
    V_sd_all{s} = V_sd;
    theta_discrete_all{s} = theta_discrete;

    % Bootstrap uncertainty (RBF-based)
    [gamma_lower, gamma_upper, gamma_resample_all] = bootstrap_uncertainty_RBF(t, ik, V_sd, lambda, n, dt, dur, OCV, R0, num_resamples, rbf_type, shape_param);

    gamma_lower_all{s} = gamma_lower;
    gamma_upper_all{s} = gamma_upper;
end

%% Plot Results
figure('Name', [AS_name, ' Type ', selected_type, ': DRT Comparison with Uncertainty'], 'NumberTitle', 'off');
hold on;
for s = 1:num_scenarios
    theta_s = theta_discrete_all{s};
    gamma_s = gamma_est_all{s};
    gamma_low_s = gamma_lower_all{s};
    gamma_upp_s = gamma_upper_all{s};

    % neg error = gamma_s - gamma_low_s
    % pos error = gamma_upp_s - gamma_s
    errorbar(theta_s, gamma_s, gamma_s - gamma_low_s, gamma_upp_s - gamma_s, ...
        '--', 'LineWidth', 1.5, 'Color', c_mat(s, :), 'DisplayName', ['Scenario ', num2str(SN_list(s))]);
end
plot(theta_true, gamma_discrete_true, 'k-', 'LineWidth', 2, 'DisplayName', 'True \gamma');
hold off;
xlabel('\theta = ln(\tau [s])', 'FontSize', labelFontSize);
ylabel('\gamma', 'FontSize', labelFontSize);
title([AS_name, ' Type ', selected_type, ': Estimated \gamma with Uncertainty'], 'FontSize', titleFontSize);
set(gca, 'FontSize', axisFontSize);
legend('Location', 'Best', 'FontSize', legendFontSize);
ylim([0 inf])

disp('Available scenario numbers:');
disp(SN_list);
selected_scenarios = input('Enter scenario numbers to plot (e.g., [1,2,3]): ');

figure('Name', [AS_name, ' Type ', selected_type, ': Selected Scenarios DRT Comparison with Uncertainty'], 'NumberTitle', 'off');
hold on;
for idx_s = 1:length(selected_scenarios)
    scn = selected_scenarios(idx_s);
    s = find(SN_list == scn);
    if ~isempty(s)
        theta_s = theta_discrete_all{s};
        gamma_s = gamma_est_all{s};
        gamma_low_s = gamma_lower_all{s};
        gamma_upp_s = gamma_upper_all{s};
        errorbar(theta_s, gamma_s, gamma_s - gamma_low_s, gamma_upp_s - gamma_s, ...
            '--', 'LineWidth', 1.5, 'Color', c_mat(s, :), 'DisplayName', ['Scenario ', num2str(SN_list(s))]);
    else
        warning('Scenario %d not found in the data', scn);
    end
end
plot(theta_true, gamma_discrete_true, 'k-', 'LineWidth', 2, 'DisplayName', 'True \gamma');
hold off;
xlabel('\theta = ln(\tau [s])', 'FontSize', labelFontSize);
ylabel('\gamma', 'FontSize', labelFontSize);
title([AS_name, ' Type ', selected_type, ': Estimated \gamma with Uncertainty for Selected Scenarios'], 'FontSize', titleFontSize);
set(gca, 'FontSize', axisFontSize);
legend('Location', 'Best', 'FontSize', legendFontSize);
ylim([0 inf])

figure('Name', [AS_name, ' Type ', selected_type, ': Individual Scenario DRTs'], 'NumberTitle', 'off');
num_cols = 5;
num_rows = ceil(num_scenarios / num_cols);

for s = 1:num_scenarios
    subplot(num_rows, num_cols, s);
    theta_s = theta_discrete_all{s};
    gamma_s = gamma_est_all{s};
    gamma_low_s = gamma_lower_all{s};
    gamma_upp_s = gamma_upper_all{s};
    errorbar(theta_s, gamma_s, gamma_s - gamma_low_s, gamma_upp_s - gamma_s, ...
        'LineWidth', 1.0, 'Color', c_mat(s, :));
    hold on;
    plot(theta_true, gamma_discrete_true, 'k-', 'LineWidth', 1.5);
    hold off;
    xlabel('\theta', 'FontSize', labelFontSize);
    ylabel('\gamma', 'FontSize', labelFontSize);
    title(['Scenario ', num2str(SN_list(s))], 'FontSize', titleFontSize);
    set(gca, 'FontSize', axisFontSize);
    ylim([0 inf])
end


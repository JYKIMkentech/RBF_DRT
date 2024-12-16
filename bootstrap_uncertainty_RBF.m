function [gamma_lower, gamma_upper, gamma_resample_all] = bootstrap_uncertainty_RBF(t, ik, V_sd, lambda, n, dt, dur, OCV, R0, num_resamples, rbf_type, shape_param)
    % bootstrap_uncertainty_RBF: Performs bootstrap to estimate uncertainty of gamma_est using RBF-based DRT estimation.
    %
    % Inputs and Outputs similar to bootstrap_uncertainty but uses RBF.

    if isscalar(dt)
        dt = repmat(dt, length(t), 1);
    end

    % Original estimation to get baseline gamma
    [gamma_est, ~, theta_discrete, ~, ~, ~] = DRT_estimation_RBF(t, ik, V_sd, lambda, n, dt, dur, OCV, R0, rbf_type, shape_param);
    gamma_resample_all = zeros(length(gamma_est), num_resamples);

    % Bootstrap
    N = length(t);
    for b = 1:num_resamples
        % Resample with replacement
        idx = randi(N, N, 1);
        t_b = t(idx);
        ik_b = ik(idx);
        V_sd_b = V_sd(idx);
        dt_b = dt(idx);

        try
            [gamma_b, ~, ~, ~, ~, ~] = DRT_estimation_RBF(t_b, ik_b, V_sd_b, lambda, n, dt_b, dur, OCV, R0, rbf_type, shape_param);
            gamma_resample_all(:, b) = gamma_b;
        catch
            % If quadprog fails, fill with NaN
            gamma_resample_all(:, b) = NaN;
        end
    end

    % Compute percentile bounds
    gamma_sorted = sort(gamma_resample_all, 2);
    lower_idx = max(round(0.025 * num_resamples),1); 
    upper_idx = min(round(0.975 * num_resamples),num_resamples);

    gamma_lower = gamma_sorted(:, lower_idx);
    gamma_upper = gamma_sorted(:, upper_idx);

    % If some NaN occurred, handle it gracefully
    nan_rows = any(isnan(gamma_sorted), 2);
    gamma_lower(nan_rows) = gamma_est(nan_rows);
    gamma_upper(nan_rows) = gamma_est(nan_rows);
end

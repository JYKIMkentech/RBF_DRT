function [gamma_est, V_est, theta_discrete, tau_discrete, W, y] = DRT_estimation_RBF(t, ik, V_sd, lambda, n, dt, ~, OCV, R0, rbf_type, shape_param)
    % DRT_estimation_RBF: RBF-based DRT estimation
    %
    % Inputs:
    %   t, ik, V_sd, lambda, n, dt, OCV, R0 : same as before
    %   rbf_type    : 'gaussian' etc.
    %   shape_param : shape parameter for RBF
    %
    % Outputs:
    %   gamma_est   : Reconstructed gamma from RBF coefficients
    %   V_est       : Estimated voltage
    %   theta_discrete, tau_discrete : discretization
    %   W, y        : system matrices/vectors

    if isscalar(dt)
        dt = repmat(dt, length(t), 1);
    elseif length(dt) ~= length(t)
        error('Length of dt must be equal to length of t if dt is a vector.');
    end

    tau_min = 0.1;
    tau_max = 1000;
    theta_min = log(tau_min);
    theta_max = log(tau_max);

    theta_discrete = linspace(theta_min, theta_max, n)';
    delta_theta = theta_discrete(2) - theta_discrete(1);
    tau_discrete = exp(theta_discrete);

    % Define RBF
    switch lower(rbf_type)
        case 'gaussian'
            rbf_func = @(theta, center) exp(- (shape_param^2)*(theta - center).^2);
        otherwise
            error('Unknown rbf_type. Implement other RBFs if needed.');
    end

    theta_centers = theta_discrete;

    % Construct Q
    Q = zeros(length(t), n);
    for k_idx = 1:length(t)
        if k_idx == 1
            for i = 1:n
                Q(k_idx, i) = ik(k_idx) * (1 - exp(-dt(k_idx) / tau_discrete(i))) * delta_theta;
            end
        else
            for i = 1:n
                Q(k_idx, i) = Q(k_idx - 1, i)*exp(-dt(k_idx)/tau_discrete(i)) + ...
                              ik(k_idx)*(1 - exp(-dt(k_idx)/tau_discrete(i)))*delta_theta;
            end
        end
    end

    % Construct Phi (RBF matrix)
    Phi = zeros(n, n);
    for m = 1:n
        Phi(:, m) = rbf_func(theta_discrete, theta_centers(m));
    end

    W = Q * Phi;  % Now gamma(theta) = Phi*x, and V = OCV+R0i+W*x

    y = V_sd - OCV - R0*ik;

    % Regularization matrix L (on x)
    L = zeros(n-1, n);
    for i = 1:n-1
        L(i, i) = -1;
        L(i, i+1) = 1;
    end

    H = 2 * (W' * W + lambda * (L' * L));
    f = -2 * W' * y;

    A_ineq = -eye(n);
    b_ineq = zeros(n, 1);

    options = optimoptions('quadprog', 'Display', 'off');
    x_coeff = quadprog(H, f, A_ineq, b_ineq, [], [], [], [], [], options);

    gamma_est = Phi * x_coeff;
    V_est = OCV + R0 * ik + W * x_coeff;
end

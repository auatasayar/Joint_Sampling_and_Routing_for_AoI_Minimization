rng("default")

Y_cap = ; %Select the delay value at which the penalty function is capped.
alpha = ; %Normalization parameter for the penalty function (if needed).

% Select the parameters of your penalty function f(t) here.
% Define the integrated version F(t) corresponding to f(t).
% Below is an example function.
f_func = @(t) alpha*(exp(min(t,Y_cap))-1); 
F_orig = @(t) alpha*(exp(t) - t);
f_val_at_cap = alpha*(exp(Y_cap) - 1); %Only use if there is a cap.
F_func = @(t) F_orig(min(t, Y_cap)) + f_val_at_cap .* max(0, t - Y_cap);

mu_r    = []; % Select mean delays.
sigma_r = []; % Select delay std's.
dist_r  = []; % Select route delay distributions (gamma/log-normal)

epsilon = 1e-5;  
max_iterations = 2000;

num_samples = 4000000; 
Y_samples = generate_samples(mu_r, sigma_r, dist_r, num_samples);
mean_Y = mean(Y_samples); 

fprintf('Pre-computing Expectation Lookup Tables...\n');
grid_max = 500; 
d_grid = [0:0.1:Y_cap, Y_cap+0.2:0.5:grid_max]'; 
Exp_F_Table = zeros(length(d_grid), length(mu_r));
for r = 1:length(mu_r)
    for i = 1:length(d_grid)
        Exp_F_Table(i, r) = mean(F_func(d_grid(i) + Y_samples(:,r)));
    end
end
fprintf('Starting Nonlinear AoI Optimization...\n');
[g_opt, G_final, t_star_final] = find_g_opt_and_G(mu_r, epsilon, max_iterations, f_func, F_func, Y_samples, mean_Y, d_grid, Exp_F_Table);

Delta = 0.01:0.1:500; 
Y_samples_plot = Y_samples(1:min(5000, end), :); 
q = zeros(length(mu_r), length(Delta));
fprintf('Computing Q curves for plotting...\n');
for i=1:length(mu_r)
    const_term_case1 = interp1(d_grid, Exp_F_Table(:,i), t_star_final(i), 'linear', 'extrap');
    
    q(i,:) = compute_Q_Interpolated(Delta, g_opt, G_final(i), t_star_final(i), ...
             Y_samples_plot(:,i), F_func, mean_Y(i), const_term_case1, ...
             d_grid, Exp_F_Table(:,i));
end
[v, ~] = min(q);
figure;
plot(Delta, q, 'LineWidth', 1.5); hold on;
plot(Delta, v, 'k--', 'LineWidth', 2);
xlabel('Age \delta'); ylabel('Q(\delta)');
title(['Nonlinear AoI: Optimal Cost c = ' num2str(g_opt)]);
grid on;
fprintf('Optimization Complete. Optimal Cost c = %.6f\n', g_opt);
for r=1:length(mu_r)
    fprintf('Route %d: Waiting threshold t* = %.4f\n', r, t_star_final(r));
end

fprintf('\n--- Routing Thresholds ---\n');
switch_indices = find(diff(min_idx) ~= 0);

if isempty(switch_indices)
    fprintf('No crossovers detected. One route is optimal.\n');
else
    for k = 1:length(switch_indices)
        idx = switch_indices(k);
        r1 = min_idx(idx); 
        r2 = min_idx(idx+1);
        x1 = Delta(idx); x2 = Delta(idx+1);
        y1_r1 = q(r1, idx); y2_r1 = q(r1, idx+1);
        y1_r2 = q(r2, idx); y2_r2 = q(r2, idx+1);
        m1 = (y2_r1 - y1_r1) / (x2 - x1);
        m2 = (y2_r2 - y1_r2) / (x2 - x1);
        tau = x1 + (y1_r2 - y1_r1) / (m1 - m2);
        
        fprintf('Switch from Route %d to Route %d at tau = %.4f\n', r1, r2, tau);
    end
end

function [g_opt, G_final, t_star_final] = find_g_opt_and_G(mu_r, epsilon, max_iterations, f_func, F_func, Y_samples, mean_Y, d_grid, Exp_F_Table)
    tol = 1e-5;
    max_bisection_iter = 15;
    left = ; right = ; %Select appropriate values (can use the formula provided in the paper).
    
    for iter = 1:max_bisection_iter
        mid = (left + right) / 2;
        [h_mid, G_values, t_star_vals] = compute_h_and_G(mid, mu_r, epsilon, max_iterations, f_func, F_func, Y_samples, mean_Y, d_grid, Exp_F_Table);
        
        if abs(h_mid) < tol, break; end
        
        if h_mid > 0
            left = mid;
        else
            right = mid;
        end
        fprintf('Bisection Iter %d: c = %.4f, h(c) = %.6f\n', iter, mid, h_mid);
    end
    g_opt = mid; G_final = G_values; t_star_final = t_star_vals;
end
function [h, G_final, t_star_vals] = compute_h_and_G(c, mu_r, epsilon, max_iterations, f_func, F_func, Y_samples, mean_Y, d_grid, Exp_F_Table)
    num_states = length(mu_r);
    t_star_vals = zeros(num_states, 1);
    const_term_case1 = zeros(num_states, 1);
    
    for r = 1:num_states
        obj_fun = @(t) mean(f_func(t + Y_samples(:,r))) - c;
        try
            if obj_fun(0) > 0
                t_star_vals(r) = 0;
            else
                t_star_vals(r) = fzero(obj_fun, max(0.1, mu_r(r))); 
            end
        catch
            t_star_vals(r) = 0; 
        end
        
        const_term_case1(r) = interp1(d_grid, Exp_F_Table(:,r), t_star_vals(r), 'linear', 'extrap');
    end
    G_curr = zeros(1, num_states);
    h_curr = 0;
    
    for K = 1:max_iterations
        G_next = zeros(1, num_states);
        
        all_Ws = compute_expected_W_Interpolated(c, G_curr, h_curr, Y_samples, ...
                                                 mu_r, t_star_vals, ...
                                                 F_func, mean_Y, const_term_case1, ...
                                                 d_grid, Exp_F_Table);
        G_next = all_Ws;
        q0s = zeros(num_states, 1);
        for r = 1:num_states
            q0s(r) = compute_Q_Interpolated(0, c, G_next(r), t_star_vals(r), ...
                                          Y_samples(:,r), F_func, mean_Y(r), const_term_case1(r), ...
                                          d_grid, Exp_F_Table(:,r));
        end
        h_next = min(q0s);
        
        if abs(h_next - h_curr) < epsilon, break; end
        h_curr = h_next; G_curr = G_next;
    end
    h = h_next; G_final = G_curr;
end
function expected_w_vec = compute_expected_W_Interpolated(c, G_prev, h_prev, Y_samples, ...
                                                          mu_r, t_star_vals, ...
                                                          F_func, mean_Y, const_term_case1, ...
                                                          d_grid, Exp_F_Table)
    num_states = length(mu_r);
    num_samples = size(Y_samples, 1);
    expected_w_vec = zeros(1, num_states);
    
    for r_curr = 1:num_states
        curr_samples = Y_samples(:, r_curr);
        
        chunk_size = 5000;
        sum_min_Q = 0;
        
        for start_idx = 1:chunk_size:num_samples
            end_idx = min(start_idx + chunk_size - 1, num_samples);
            block_Y = curr_samples(start_idx:end_idx);
            
            block_Q = zeros(length(block_Y), num_states);
            
            for r_next = 1:num_states
                block_Q(:, r_next) = compute_Q_Interpolated(block_Y, c, G_prev(r_next), ...
                                         t_star_vals(r_next), [], F_func, mean_Y(r_next), ...
                                         const_term_case1(r_next), d_grid, Exp_F_Table(:,r_next));
            end
            
            sum_min_Q = sum_min_Q + sum(min(block_Q, [], 2));
        end
        
        expected_w_vec(r_curr) = (sum_min_Q / num_samples) - h_prev;
    end
end
function q = compute_Q_Interpolated(delta, c, G, t_star, ~, F_func, mean_Y, const_term_case1, d_grid, Exp_F_Table)
    delta = delta(:);
    q = zeros(size(delta));
    term2 = F_func(delta);
    
    idx1 = delta <= t_star;
    if any(idx1)
        z_star = t_star - delta(idx1);
        q(idx1) = const_term_case1 - term2(idx1) - c * (z_star + mean_Y) + G;
    end
    
    idx2 = ~idx1;
    if any(idx2)
        d_vals = delta(idx2);
        
        mean_F = interp1(d_grid, Exp_F_Table, d_vals, 'linear', 'extrap');
        
        q(idx2) = mean_F - term2(idx2) - c * mean_Y + G;
    end
end
function Y_samples = generate_samples(mu_r, sigma_r, dist_r, num_samples)
    num_states = length(mu_r);
    Y_samples = zeros(num_samples, num_states);
    for r = 1:num_states
        if dist_r(r) == 'l'
            mu_l = log((mu_r(r)^2)/sqrt(sigma_r(r)^2+mu_r(r)^2));
            sigma_l = sqrt(log(sigma_r(r)^2/(mu_r(r)^2)+1));
            Y_samples(:,r) = lognrnd(mu_l, sigma_l, [num_samples, 1]);
        elseif dist_r(r) == 'g'
            if sigma_r(r) > 1e-6
                b = (sigma_r(r)^2)/mu_r(r);
                a = mu_r(r)/b;
                Y_samples(:,r) = gamrnd(a, b, [num_samples, 1]);
            else
                Y_samples(:,r) = mu_r(r) * ones(num_samples, 1);
            end
        end
    end
end

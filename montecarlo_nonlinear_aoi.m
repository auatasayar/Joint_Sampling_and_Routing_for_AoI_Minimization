%% AoI Monte Carlo (Simple case of two routes, one routing threshold, with capped exponential penalty)
rng("default")
%Parameters
num_cycles = 10^8;      
alpha = 1; % Exponential parameter          
y_cap = ; % Cap delay           
tau = ; % Single threshold between the routes.


% Route-specific waiting thresholds
wait_thresh_log = ; 
wait_thresh_gam = ;
% Route 1: Log-normal
m1 = ; v1 = ; % Select mean and variance
mu_log = log(m1^2 / sqrt(v1 + m1^2));
sigma_log = sqrt(log(v1/m1^2 + 1));

% Route 2: Gamma
m2 = ; v2 = ; % Select mean and variance
shape_gam = m2^2 / v2;
scale_gam = v2 / m2;

penalty_at_cap = exp(alpha * y_cap);

F = @(t) (t <= y_cap) .* ( (1/alpha) * exp(alpha * min(t, y_cap)) ) + ...
         (t > y_cap)  .* ( (1/alpha) * penalty_at_cap + (t - y_cap) * penalty_at_cap );

% --- Simulation ---
total_area = 0;
total_time = 0;
current_y = 1.0;
y_log = lognrnd(mu_log, sigma_log,[1,num_cycles]);
y_gam = gamrnd(shape_gam, scale_gam,[1,num_cycles]);

for i = 1:num_cycles
    if current_y < tau
        w = max(0, wait_thresh_log - current_y);
        y_prime = y_log(i);
    else
        w = max(0, wait_thresh_gam - current_y);
        y_prime = y_gam(i);
    end
    
    t_start = current_y;
    t_end = current_y + w + y_prime;
    
    cycle_area = F(t_end) - F(t_start);
    
    total_area = total_area + cycle_area;
    total_time = total_time + (w + y_prime);
    
    current_y = y_prime;
end

average_penalty = total_area / total_time;

fprintf('--- Results ---\n');
fprintf('Average Long-Term Penalty: %.6f\n', average_penalty);

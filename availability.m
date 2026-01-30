rng("default")

mu_r    = []; % Select mean delays.
sigma_r = []; % Select delay std's.
dist_r  = []; % Select route delay distributions (gamma/log-normal)
p_vec   = []; % Select availability probabilities (at least one should be 1)

epsilon        = 1e-3;
max_iterations = 5000;

[g_opt, G_final] = find_g_opt_and_G( ...
    mu_r, sigma_r, epsilon, max_iterations, dist_r, p_vec);

fprintf("Optimal age = %.6f\n", g_opt);
fprintf("Final G per route: [%s]\n", sprintf("%.6f ", G_final));

% Splitting to patterns to decide the routing thresholds. (For the case
% where route 1 is always available)

M = numel(mu_r);
nPatterns = 2^(M-1);
all_s = false(nPatterns, M);

for mask = 0:(nPatterns-1)
    bits = bitget(mask, 1:(M-1));   % for routes 2..M
    all_s(mask+1, :) = [1, bits];   % route1 ON, others per bits
end

Delta = 0.01:0.01:500;
numD  = numel(Delta);

Q_s = cell(nPatterns,1);

for p = 1:nPatterns
    s     = all_s(p,:);
    avail = find(s);
    Qmat  = +inf(M, numD);
    for r = avail
        for j = 1:numD
            y = Delta(j);
            Qmat(r,j) = compute_Q_K( ...
                y, r, g_opt, ...
                mu_r(r), sigma_r(r), ...
                G_final(r) );
        end
    end
    Q_s{p} = Qmat;
end

taus = cell(nPatterns,1);

for p = 1:nPatterns
    s     = all_s(p,:);
    Qmat  = Q_s{p};
    avail = find(s);

    [~, bestIdx] = min(Qmat(avail,:), [], 1);
    switches = find(diff(bestIdx)~=0);
    taus{p}   = Delta(switches);

    fprintf("Pattern s = [%s]  →  thresholds τ = [%s]\n", ...
        sprintf("%d", s), sprintf("%.2f ", taus{p}));
end
age= g_opt;

function [g_opt, G_final] = find_g_opt_and_G( ...
        mu_r, sigma_r, eps, max_iter, dist_r, p_vec)

    fprintf('--- Starting bisection on g ---\n');
    left  = 0;
    right = 2*mu_r(1);
    tol   = 1e-5;

    for iter = 1:13
        mid = (left + right)/2;
        [h_mid, G_vals] = compute_h_and_G( ...
            mid, mu_r, sigma_r, eps, max_iter, dist_r, p_vec);
        fprintf('Bisection %2d: g=%.6f, h(g)=%.6f\n', iter, mid, h_mid);

        if abs(h_mid) < tol
            g_opt   = mid;
            G_final = G_vals;
            fprintf('Converged on g_opt=%.6f after %d iters\n', g_opt, iter);
            return;
        end
        if h_mid > 0
            left = mid;
        else
            right = mid;
        end
    end

    warning('Bisection did not converge within 30 iters');
    g_opt = (left+right)/2;
    [~, G_final] = compute_h_and_G( ...
        g_opt, mu_r, sigma_r, eps, max_iter, dist_r, p_vec);
end

function [h, G_final] = compute_h_and_G( ...
        c, mu_r, sigma_r, eps, max_iter, dist_r, p_vec)

    R = length(mu_r);
    N = 300000;
    fprintf(' compute_h_and_G: solving for c=%.6f ...\n', c);
    % pre‑sample Y for each route
    Y = zeros(N,R);
    for r = 1:R
      switch dist_r(r)
        case 'z'
          Y(:,r) = randsample([0,60], N, true, [2/3,1/3]);
        case 'g'
          if sigma_r(r)>0
            b = sigma_r(r)^2/mu_r(r);
            a = mu_r(r)/b;
            Y(:,r) = gamrnd(a,b,[N,1]);
          else
            Y(:,r) = mu_r(r);
          end
        case 'l'
            mu_l = log((mu_r(r)^2)/sqrt(sigma_r(r)^2+mu_r(r)^2));
            sigma_l = sqrt(log(sigma_r(r)^2/(mu_r(r)^2)+1));
            Y(:,r) = lognrnd(mu_l, sigma_l, [N, 1]);
        otherwise
          Y(:,r) = normrnd(mu_r(r), sigma_r(r), [N,1]);
      end
    end

    h_hist = zeros(max_iter,1);
    G_hist = zeros(max_iter,R);

    % K=1 init
    Q0  = arrayfun(@(r) compute_Q_K(0,r,c,mu_r(r),sigma_r(r),0), 1:R);
    h_hist(1) = min(Q0);

    for K = 2:max_iter
      for r = 1:R
        G_hist(K,r) = compute_expected_W( ...
          K, r, c, mu_r, sigma_r, G_hist(K-1,:), h_hist(K-1), Y(:,r), p_vec);
      end
      Q0 = Q0 + (G_hist(K,:) - G_hist(K-1,:));
      h_hist(K) = min(Q0);
      if abs(h_hist(K)-h_hist(K-1))<eps
        h       = h_hist(K);
        G_final = G_hist(K,:);
        return;
      end
    end

    warning('compute_h_and_G did not converge');
    h       = h_hist(end);
    G_final = G_hist(end,:);
end

function expected_w = compute_expected_W( ...
        K, r, c, mu_r, sigma_r, G_prev, h_prev, Y_r, p_vec)
    
    N = numel(Y_r);
    R = numel(mu_r);
    Q_all = inf(N, R);
    for j = 1:R
        delta = Y_r;
        p_part = max(0, c - mu_r(j) - delta);
        Qj = -0.5*(p_part.^2) ...
             + (delta - c)*mu_r(j) ...
             + (sigma_r(j)^2 + mu_r(j)^2)/2 ...
             + G_prev(j);
        Q_all(:,j) = Qj;
    end

    S = rand(N, R) < repmat(p_vec, N, 1);
    Q_all(~S) = Inf;
    W = min(Q_all, [], 2) - h_prev;
    expected_w = mean(W);
end

function q = compute_Q_K(delta, r, c, mu, sigma, G)
    p = max(0, c - mu - delta);
    q = -p^2/2 + (delta-c)*mu + (sigma^2+mu^2)/2 + G;
end

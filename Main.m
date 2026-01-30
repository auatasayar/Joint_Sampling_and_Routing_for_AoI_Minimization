rng("default")

ages = zeros(1,10);
aoi_utku = zeros(1,10);
for loop = 1:10
    loop

% mu_r    = [10, 8, 6, 1.7, 1, 0.6, 0.1];
% sigma_r = [5, 0.5, 2, 5.1, 1, 5.5, 6.3];
% dist_r  = ['g','l','l','l','l','l','l'];
% p_vec   = [1, 0.5, 0.5, 0.5, 0.2, 0.5, 0.5]; %0.643

% mu_r    = [30,3,2,1.2,1,0.5,0.2];
% sigma_r = [0,0.2,0.3,0.5,1,2,3];
% dist_r  = ['g','l','l','l','l','l','l'];
% p_vec   = [1, 0.5, 0.6, 0.5, 0.5, 0.6, 0.5]; %0.9144


% mu_r    = [20,3,2,1.5,1,0.5];
% sigma_r = [0.2,0.5,1,2,2.5,2.9];
% dist_r  = ['g','l','l','l','l','l'];
% p_vec   = [1, 0.7, 0.6, 0.5, 0.4, 0.5]; %0.858

% mu_r    = [40,3,2,1.2,1,0.5,0.2];
% sigma_r = [0,0.2,0.3,0.5,1,2,3];
% dist_r  = ['g','l','l','l','l','l','l'];
% p_vec   = [1, 0.6, 0.7, 0.6, 0.7, 0.7, 0.6]; %0.954

% mu_r    = [40,3,2,1.2,1,0.5,0.2];
% sigma_r = [0,0.2,0.3,0.5,1,2,3];
% dist_r  = ['g','l','l','l','l','l','l'];
% p_vec_add = [0.15,0.25,0.15,0.25,0.25,0.15] + (29)*0.01;
% p_vec   = [1, p_vec_add]; %0.954

% mu_r    = [10,3];
% sigma_r = [5,4];
% dist_r  = ['g','g'];
% p_vec   = [1, 0.3]; %0.954

mu_r    = [6,5,3];
sigma_r = [2,4,7];
dist_r  = ['g','l','g'];
p_vec   = [1, (loop-1)*0.02, (loop-1)*0.02];

% mu_r    = [10,4,3]; % mean delays
% sigma_r = [8,4,6]; % std devs
% dist_r  = ['g','l','l']; % distribution types per route
% p_vec   = [1, 0.1 + (loop-1)*0.02, 0.1 + (loop-1)*0.02];

aoi_utku(loop) = (1-p_vec(2))*(1-p_vec(3))*mu_r(1) + (1-p_vec(3))*p_vec(2)*mu_r(2)+p_vec(3)*mu_r(3) + (((1-p_vec(2))*(1-p_vec(3))*mu_r(1))*(mu_r(1)/2 + sigma_r(1)^2/(2*mu_r(1))) + ...
    (1-p_vec(3))*p_vec(2)*mu_r(2)*(mu_r(2)/2 + sigma_r(2)^2/(2*mu_r(2))) + p_vec(3)*mu_r(3)*(mu_r(3)/2 + sigma_r(3)^2/(2*mu_r(3))))/ ((1-p_vec(2))*(1-p_vec(3))*mu_r(1) + (1-p_vec(3))*p_vec(2)*mu_r(2)+p_vec(3)*mu_r(3));


epsilon        = 1e-3;
max_iterations = 5000;

[g_opt, G_final] = find_g_opt_and_G( ...
    mu_r, sigma_r, epsilon, max_iterations, dist_r, p_vec);

fprintf("Optimal g = %.6f\n", g_opt);
fprintf("Final G per route: [%s]\n", sprintf("%.6f ", G_final));

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

% example_p = 1;
% figure; hold on;
% for r = find(all_s(example_p,:))
%     plot(Delta, Q_s{example_p}(r,:), 'DisplayName', sprintf('r=%d',r));
% end
% xlabel('y (previous delay)'); ylabel('Q(y,s;r)');
% title(sprintf('Q-curves for pattern s=[%s]', sprintf('%d',all_s(example_p,:))));
% legend; hold off;
ages(loop) = g_opt;
end
function [F, P, G, Q, K_bar] = MDA_quants(K, X, Y)
%{
computation of quantities in Multi-domain Discriminate Analysis (MDA)

INPUT
    K      - kernel matrix of data of all source domains
    X      - cell of (n_s*d) matrices, each matrix corresponds to the instance features of a source domain
    Y      - cell of (n_s*1) matrices, each matrix corresponds to the instance labels of a source domain

OUTPUT
    F      - F in avearge class discrepancy, Eq.(11)
    P      - P in multi-domain between class scatter, Eq.(15)
    G      - G in average domain discrepancy, Eq.(6)
    Q      - Q in multi-domain within class scatter, Eq.(18)
    K_bar  - centered kernel matrix 

Shoubo Hu (shoubo.sub [at] gmail.com)
2019-06-03
%}

% ----------------------------------------------------------------------

% number of domain
n_domain = length(X);

% labels of all domains in a vector
Y_ALL = cat(1, Y{:});
n_total = size(Y_ALL, 1);            % number of instances in all source domains
n_class = length(unique(Y_ALL));   % number of classes mush begin from one

% save class and domain index of all instances into two row vectors 
class_index = zeros(1, n_total);
domain_index = zeros(1, n_total);
count = 1;
for s = 1:n_domain
    for i = 1:size(Y{s}, 1)
        temp_c = Y{s}(i);
        class_index(count) = temp_c;
        domain_index(count) = s;
        count = count + 1;
    end
end

% count and proportion matrix
% [cnt_mat]_{sj} is the number of instances in domain s class j
cnt_mat = zeros(n_domain, n_class);
for s = 1:n_domain
    idx = find(domain_index==s);
    for j = 1:n_class
        idx_2 = idx(class_index(idx)==j);
        cnt_mat(s, j) = length(idx_2);
    end
end

% [prpt_vec]_{sj} is n^{s}_{j} / n^{s}
prpt_vec = cnt_mat ./ repmat(sum(cnt_mat, 2), 1, n_class);
sum_over_dm_vec = sum(prpt_vec, 1); % sum_{s} [ n^{s}_{j} / n^{s} ]
% ns_vec = sum(cnt_mat, 2); % col vec, ns_vec(s, 1) is the number of pts in domain s
nj_vec = sum(cnt_mat, 1); % row vec, nj_vec(1, j) is the number of pts in class j

class_domain_mean = cell(1, n_domain);
for s = 1:n_domain
    idx = find(domain_index==s);
    domain_mean = zeros(n_total, n_class);
    for j = 1:n_class
        idx_2 = idx(class_index(idx)==j);
        domain_mean(:,j) = mean(K(:, idx_2), 2);
    end
    class_domain_mean{s} = domain_mean;
end

u_j_mat = zeros(n_total, n_class);
for j = 1:n_class
    u_j = zeros(n_total, 1);
    for s = 1:n_domain
        u_j = u_j + class_domain_mean{s}(:, j) * prpt_vec(s, j);
    end
    u_j_mat(:, j) = u_j / sum_over_dm_vec(1, j);
end

% ----- compute matrix P -----
u_bar = zeros(n_total, 1);
for j = 1:n_class
    u_bar = u_bar + u_j_mat(:, j) * (sum_over_dm_vec(1, j) / n_domain);
end

P = zeros(n_total, n_total);
for j = 1:n_class
    P = P + nj_vec(1, j) * ( u_j_mat(:, j) - u_bar ) * ( ( u_j_mat(:, j) - u_bar )' );
end
P = P / n_total;


% ----- compute matrix F -----
F = zeros(n_total, n_total);
for j1 = 1:(n_class - 1)
    for j2 = (j1 + 1):n_class
        F = F + (u_j_mat(:, j1) - u_j_mat(:, j2)) * ( (u_j_mat(:, j1) - u_j_mat(:, j2))' );
    end
end
F = F / (n_class * (n_class - 1) * 0.5);

% ----- compute matrix Q -----
Q = zeros(n_total, n_total);
for j = 1:n_class
    idx = find(class_index==j);
    G_j = u_j_mat(:, j);
    G_ij = K(:, idx);
    Q_i = G_ij - repmat(G_j, 1, length(idx));
    Q = Q + Q_i * Q_i';
end
Q = Q / n_total;

% ----- compute matrix G -----
G = zeros(n_total, n_total);
for j = 1:n_class
    for s1 = 1:(n_domain - 1)
        idx = find(domain_index == s1);
        idx_2 = find(class_index(idx)==j);
        idx_2 = idx(idx_2);
        left = mean(K(:, idx_2), 2);
        for s2 = (s1 + 1):n_domain
            idx = find(domain_index == s2);
            idx_2 = find(class_index(idx)==j);
            idx_2 = idx(idx_2);
            right = mean(K(:, idx_2), 2);
            G = G + (left - right)* ((left - right)');
        end
    end
end
G = G / (n_class*n_domain*(n_domain-1)*0.5);

J = ones(n_total, n_total)*(1/n_total);
K_bar = K - J*K - K*J + J*K*J;
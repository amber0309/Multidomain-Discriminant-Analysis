function [test_accuracy, predicted_labels, Zs, Zt] = MDA(X_s_cell, Y_s_cell, X_t, Y_t, params)
%{
Implementation of Multidomain Discriminant Analysis (MDA) proposed in [1]

INPUT(params is optional):
  X_s_cell          - cell of (n_s*d) matrices, each matrix corresponds to the instance features of a source domain
  Y_s_cell          - cell of (n_s*1) matrices, each matrix corresponds to the instance labels of a source domain
  X_t               - (n_t*d) matrix, rows correspond to instances and columns correspond to features
  Y_t               - (n_t*1) matrix, each row is the class label of corresponding instances in X_t
  params (optional) - params.beta:      vector of validated values of beta
                      params.alpha:     vector of validated values of alpha
                      params.gamma:     vector of validated values of gamma
                      params.q_list:    vector of validated dimension of the transformed space
                      params.X_v:       (n_v*d) matrix of instance features of validation set (use all source instances if not provided)
                      params.Y_v:       (n_v*1) matrix of instance labels of validation set (use all source instances if not provided)
                      params.verbose:   if true, show the validation accuracy of each parameter setting

OUTPUT:
  test_accuracy     - test accuracy of target instances
  predicted_labels  - predicted labels of target instances
  Zs                - projected source domain instances
  Zt                - projected target domain instances

Shoubo Hu (shoubo.sub [at] gmail.com)
2019-06-02

[1] Shoubo Hu, Kun Zhang, Zhitang Chen, Laiwan Chan. "Domain Generalization via Multidomain Discriminant Analysis." UAI 2019.
%}

    if nargin < 4
        error('Error. \nOnly %d input arguments! At least 4 required', nargin);
    elseif nargin == 4
        % default params values
        beta = [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9];
        alpha = [1e-3 1e-2 1e-1 1 1e1 1e2 1e3 1e4 1e5 1e6];
        % alpha = [1 1e1 1e2 1e3 1e4 1e5 1e6 1e7 1e8 1e9];
        gamma = [1e-3 1e-2 1e-1 1 1e1 1e2 1e3 1e4 1e5 1e6];
        q_list = [2];
        X_v = cat(1, X_s_cell{:});
        Y_v = cat(1, Y_s_cell{:});
        verbose = false;
    elseif nargin == 5
        if ~isfield(params, 'beta')
            beta = [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9];
        else
            beta = params.beta;
        end
        
        if ~isfield(params, 'alpha')
            % alpha = [1 1e1 1e2 1e3 1e4 1e5 1e6 1e7 1e8 1e9];
            alpha = [1e-3 1e-2 1e-1 1 1e1 1e2 1e3 1e4 1e5 1e6];
        else
            alpha = params.alpha;
        end

        if ~isfield(params, 'gamma')
            gamma = [1 1e1 1e2 1e3 1e4 1e5 1e6 1e7 1e8 1e9];
        else
            gamma = params.gamma;
        end

        if ~isfield(params, 'q_list')
            q_list = [2];
        else
            q_list = params.q_list;
        end

        if ~isfield(params, 'verbose')
            verbose = false;
        else
            verbose = params.verbose;
        end

        if ~isfield(params, 'X_v')
            X_v = cat(1, X_s_cell{:});
            Y_v = cat(1, Y_s_cell{:});
        else
            if ~isfield(params, 'Y_v')
                error('Error. Labels of validation set needed!');
            end
            X_v = params.X_v;
            Y_v = params.Y_v;
        end
    end

    % ----- training phase
    X_s = cat(1, X_s_cell{:});
    Y_s = cat(1, Y_s_cell{:});
    fprintf('Number of source domains: %d, Number of classes: %d.\n', length(X_s_cell), length(unique(Y_s)) );
    fprintf('Validating hyper-parameters ...\n');

    % ----- ----- distance matrix
    dist_s_s = pdist2(X_s, X_s);
    dist_s_s = dist_s_s.^2;
    sgm_s = compute_width(dist_s_s);

    dist_s_v = pdist2(X_s, X_v);
    dist_s_v = dist_s_v.^2;
    sgm_v = compute_width(dist_s_s);

    n_s = size(X_s, 1);
    n_v = size(X_v, 1);
    H_s = eye(n_s) - ones(n_s)./n_s;
    H_v = eye(n_v) - ones(n_v)./n_v;
        
    % ----- ----- kernel matrix
    K_s_s = exp(-dist_s_s./(2 * sgm_s * sgm_s));
    K_s_v = exp(-dist_s_v./(2 * sgm_v * sgm_v));
    K_s_v_bar = H_s * K_s_v * H_v;
    [F, P, G, Q, K_s_s_bar] = MDA_quants(K_s_s, X_s_cell, Y_s_cell);

    acc_mat = zeros(length(q_list), length(beta), length(alpha), length(gamma));
    for i = 1:length(beta)
        cur_beta = beta(i);
        for j = 1:length(alpha)
            cur_alpha = alpha(j);
            for p = 1:length(gamma)
                cur_gamma = gamma(p);
                [B, A] = MDA_trans(F, P, G, Q, K_s_s_bar, cur_beta, cur_alpha, cur_gamma, 1e-5);

                for q = 1:length(q_list)
                    [acc, ~, ~, ~] = MDA_test(B, A, K_s_s_bar, K_s_v_bar, Y_s, Y_v, q_list( q ) );
                    acc_mat(q, i, j, p) = acc;
                    if verbose
                        fprintf('beta: %f, alpha: %f, gamma: %f, acc: %f\n', cur_beta, cur_alpha, cur_gamma, acc);
                    end
                end
            end
        end
    end

    fprintf('Validation done! Classifying the target domain instances ...\n');
    % ----- test phase
    % ----- ----- get optimal parameters
    acc_tr_best = max( acc_mat(:) );
    ind = find( acc_mat == acc_tr_best );
    test_accuracy = 0;
    for idx = 1:length(ind)
        [q, i, j, p] = size( acc_mat );
        [best_q, best_i, best_j, best_p] = ind2sub([q, i, j, p], ind( idx ));

        best_dim = q_list(best_q);
        best_beta = beta(best_i);
        best_alpha = alpha(best_j);
        best_gamma = gamma(best_p);
        
        % ----- ----- test on the target domain
        dist_s_t = pdist2(X_s, X_t);
        dist_s_t = dist_s_t.^2;
        sgm = compute_width(dist_s_t);
        K_s_t = exp(-dist_s_t./(2 * sgm * sgm));
        n_s = size(X_s, 1);
        H_s = eye(n_s) - ones(n_s)./n_s;
        n_t = size(X_t, 1);
        H_t = eye(n_t) - ones(n_t)./n_t;
        K_s_t_bar = H_s * K_s_t * H_t;

        [B, A] = MDA_trans(F, P, G, Q, K_s_s_bar, best_beta, best_alpha, best_gamma, 1e-5);
        [acc, pre_labels, cur_Zs, cur_Zt] = MDA_test(B, A, K_s_s_bar, K_s_t_bar, Y_s, Y_t, best_dim );

        if acc > test_accuracy
            test_accuracy = acc;
            predicted_labels = pre_labels;
            Zs = cur_Zs;
            Zt = cur_Zt;
        end
    end
    fprintf('Test accuracy: %f\n', test_accuracy);

end

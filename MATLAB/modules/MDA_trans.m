function [B, A] = MDA_trans(F, P, G, Q, K_bar, beta, aph, gamma, epsilon)
%{
Compute the transformation in Multi-domain Discriminate Analysis (MDA)

INPUT
    F                   - F in avearge class discrepancy, Eq.(11)
    P                   - P in multi-domain between class scatter, Eq.(15)
    G                   - G in average domain discrepancy, Eq.(6)
    Q                   - Q in multi-domain within class scatter, Eq.(18)
    K_bar               - centered kernel matrix
    beta, aph, gamma    - trade-off parameters in Eq.(20)
    epsilon             - a small constant for numerical stability

OUTPUT
    B                   - matrix of projection
    A                   - corresponding eigenvalues

Shoubo Hu (shoubo.sub [at] gmail.com)
2019-06-03
%}

% ----------------------------------------------------------------------

    I_0 = eye( size(K_bar) );
    F1 = beta * F + (1 - beta) * P;
    F2 = ( gamma * G + aph * Q + K_bar + epsilon * I_0);
    F2_inv_F1 = F2\F1;
    
    [B, A] = eig(F2_inv_F1);
    B = real(B);
    A = real(A);
    eigvalues = diag(A);
    [val, idx] = sort(eigvalues, 'descend');
    B = B(:, idx);
    A= diag(val);

end
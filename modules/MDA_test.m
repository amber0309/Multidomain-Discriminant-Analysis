function [acc, pre_labels, Zs, Zt] = MDA_test(B, A, K_s, K_t, Y_s, Y_t, eig_ratio)
%{
Apply transformation learned by Multi-domain Discriminate Analysis (MDA) on target instances

INPUT:
  B           - transformation matrix
  A           - eigenvalues
  K_s         - (n*n) kernel matrix of all source instances
  K_t         - (n*n_t) kernel matrix between source and target instances 
  Y_s         - (n*1) matrix of class labels of all source instances 
  Y_t         - (n_t*1) matrix of class labels of target instances
  eig_ratio   - dimension of the transformed space

OUTPUT:
  acc         - test accuracy of target instances
  pre_labels  - predicted labels of target instances
  Zs          - projected all source instances
  Zt          - projected target instances

Shoubo Hu (shoubo.sub [at] gmail.com)
2019-06-03
%}

% ----------------------------------------------------------------------

vals = diag(A);
ratio = [];
count = 0;
for i = 1:length(vals)
    if vals(i)<0
        break;
    end
    count = count + vals(i);
    ratio = [ratio; count];
    vals(i) = 1/sqrt(vals(i));
end
A_sqrt = diag(vals);
ratio = ratio/count;

if eig_ratio <= 1
    idx = find(ratio>eig_ratio);
    n_eigs = idx(1);
else
    n_eigs = eig_ratio;
end

Zt = K_t' * B(:, 1:n_eigs) * A_sqrt(1:n_eigs, 1:n_eigs);

Zs = K_s' * B(:, 1:n_eigs) * A_sqrt(1:n_eigs, 1:n_eigs);

Mdl = fitcknn(Zs, Y_s);
pre_labels = predict(Mdl, Zt);
acc = length(find(pre_labels == Y_t)) / length(pre_labels);
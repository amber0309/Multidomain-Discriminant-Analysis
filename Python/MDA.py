"""
Python implementation of Multidomain discriminant analysis (MDA)
(tested on Anaconda 5.3.0 64-bit for python 2.7.15 on Windows 10)

Shoubo Hu (shoubo.sub AT gmail.com)
2019-08-13
"""
from __future__ import division
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist

class MDA():
	"""docstring for MDA"""
	def __init__(self, X_s_list, y_s_list, params=None):
		self.X_s_list = X_s_list
		self.y_s_list = y_s_list
		self.X_s = np.concatenate(X_s_list)
		self.y_s = np.concatenate(y_s_list)

		self.n_domain = len(self.X_s_list)
		self.n_total = self.X_s.shape[0]
		self.n_class = len( np.unique(self.y_s) )
		print('Number of source domains: {}'.format(self.n_domain) )
		print('Number of classes: {}'.format(self.n_class ) )

		self.dist_s_s = cdist(self.X_s, self.X_s)
		self.dist_s_s = self.dist_s_s**2
		self.sgm_s = compute_width(self.dist_s_s)
		self.K_s_s = np.exp( -self.dist_s_s / (2 * self.sgm_s * self.sgm_s) )

		n_s = self.X_s.shape[0]
		self.H_s = np.eye(n_s, dtype = float) - np.ones((n_s, n_s), dtype = float) / n_s

		self.params = params
		if 'verbose' not in self.params:
			self.params['verbose'] = False

		self._quantities(self.K_s_s)

	def _quantities(self, K):
		"""
		Compute quantities in Multidomain Discriminant Analysis (MDA)

		INPUT
		    K 		 kernel matrix of data of all source domains

		OUTPUT
		    F 		 F in avearge class discrepancy, Eq.(11)
		    P 		 P in multi-domain between class scatter, Eq.(15)
		    G 		 G in average domain discrepancy, Eq.(6)
		    Q 		 Q in multi-domain within class scatter, Eq.(18)
		    K_bar 	 the kernel matrix (K)
		"""

		# save class and domain index of all obs into two row vectors 
		class_index = np.zeros((self.n_total,), dtype = float)
		domain_index = np.zeros((self.n_total,), dtype = float)
		count = 0
		for s in range(0, self.n_domain):
			for i in range(0, self.y_s_list[s].shape[0]):
				temp_c = self.y_s_list[s][i]
				class_index[count] = temp_c
				domain_index[count] = s
				count = count + 1

		# prepare count and proportion matrix
		# cnt_mat_{sj} is the number of pts in domain s class j
		cnt_mat = np.zeros((self.n_domain, self.n_class), dtype = float)
		domain_index_list = []
		class_index_list = []
		for s in range(0, self.n_domain):
			idx_s = np.where( domain_index == s )[0]
			domain_index_list.append(idx_s)

			for j in range(0, self.n_class):
				if s == 0:
					idx_j = np.where( class_index == j )[0]
					class_index_list.append(idx_j)
				else:
					idx_j = class_index_list[j]

				idx = np.intersect1d(idx_s, idx_j, assume_unique = True)
				cnt_mat[s, j] = len(idx)

		# [prpt_vec]_{sj} is n^{s}_{j} / n^{s}
		prpt_vec = cnt_mat * np.reciprocal( np.tile( np.sum(cnt_mat, axis = 1).reshape(-1, 1), (1, self.n_class)) )
		sum_over_dm_vec = np.sum( prpt_vec, axis = 0 )
		nj_vec = np.sum( cnt_mat, axis = 0 )

		class_domain_mean = [None for _ in range(self.n_domain)]
		for s in range(0, self.n_domain):
			idx_s = domain_index_list[s]
			domain_mean = np.zeros((self.n_total, self.n_class), dtype = float)

			for j in range(0, self.n_class):
				idx_j = class_index_list[j]
				idx = np.intersect1d(idx_s, idx_j, assume_unique = True)
				domain_mean[:,j] = np.mean( K[:, idx], axis = 1 )

			class_domain_mean[s] = domain_mean

		u_j_mat = np.zeros( (self.n_total, self.n_class ), dtype = float)
		for j in range(0, self.n_class):
			u_j = np.zeros( (self.n_total, 1), dtype=float)
			for s in range(0, self.n_domain):
				u_j = u_j + class_domain_mean[s][:,j].reshape(-1,1) * prpt_vec[s, j]
			u_j_mat[:,j] = u_j[:,0] / sum_over_dm_vec[j]

		# compute matrix P
		u_bar = np.zeros( (self.n_total, ), dtype = float)
		for j in range(0, self.n_class):
			u_bar = u_bar + u_j_mat[:,j] * ( sum_over_dm_vec[j] / self.n_domain )

		pre_P = np.zeros( (self.n_total, self.n_total ), dtype = float)
		for j in range(0, self.n_class):
			diff = (u_j_mat[:,j]-u_bar).reshape(-1,1)
			pre_P = pre_P + nj_vec[j] * np.dot( diff, diff.T )

		P = pre_P / self.n_total

		# compute matrix F
		F = np.zeros( (self.n_total, self.n_total ), dtype = float)
		for j1 in range(0, self.n_class - 1):
			for j2 in range( j1+1, self.n_class ):
				temp = u_j_mat[:, j1].reshape(-1,1) - u_j_mat[:, j2].reshape(-1,1)
				F = F + np.dot( temp, temp.T )

		F = F / (self.n_class * (self.n_class - 1) * 0.5)

		# compute matrix Q
		Q = np.zeros((self.n_total, self.n_total), dtype = float)
		for j in range(0, self.n_class):
			idx_j = class_index_list[j]
			G_j = u_j_mat[:,j].reshape(-1,1)

			G_ij = K[:, idx_j]
			Q_i = G_ij - np.tile( G_j, (1, len(idx_j)) )

			Q = Q + np.dot( Q_i, Q_i.T )

		Q = Q / self.n_total

		# compute matrix G
		G = np.zeros((self.n_total, self.n_total), dtype = float)
		for j in range(0, self.n_class):
			for s1 in range(0, self.n_domain - 1):
				idx = np.intersect1d( domain_index_list[s1], class_index_list[j], assume_unique = True )
				left = np.mean( K[:,idx], axis=1 ).reshape(-1,1)

				for s2 in range(s1+1, self.n_domain):
					idx = np.intersect1d( domain_index_list[s2], class_index_list[j], assume_unique = True )
					right = np.mean(K[:,idx], axis=1 ).reshape(-1,1)
					temp = left - right
					G = G + np.dot( temp, temp.T )

		G = G / (self.n_class * self.n_domain * (self.n_domain-1) * 0.5)

		J = np.ones( (self.n_total, self.n_total), dtype=float ) / self.n_total
		K_bar = K - np.dot(J, K) - np.dot(K, J) + np.dot(np.dot(J, K), J)
		self.F, self.P, self.G, self.Q, self.K_s_s_bar = F, P, G, Q, K_bar

	def _transformation(self, beta, alph, gamm, eps=1e-5):
		"""
		compute the transformation in Multidomain Discriminant Analysis (MDA)

		INPUT
		    beta, alph, gamm    - trade-off parameters in Eq.(20)
		    eps                 - coefficient of the identity matrix (footnote in page 5)

		OUTPUT
		    B                   - matrix of projection
		    A                   - corresponding eigenvalues
		"""

		I_0 = np.eye( self.n_total )
		F1 = beta * self.F + (1 - beta) * self.P
		F2 = ( gamm * self.G + alph * self.Q + self.K_s_s_bar + eps * I_0 )
		F2_inv_F1 = np.linalg.solve(F2, F1)

		[A, B] = np.linalg.eig( F2_inv_F1 )
		B, A = np.real(B), np.real(A)
		idx = np.argsort(A)
		idx = np.flip(idx, axis = 0)
		A = A[idx]
		B = B[:, idx]
		A = np.diag(A)

		return B, A

	def _test(self, B, A, K_t, y_t, eig_ratio):
		"""
		Apply transformation learned by Multidomain Discriminant Analysis (MDA) on test data

		INPUT:
		  B           - transformation matrix
		  A           - eigenvalues
		  K_t         - kernel matrix of target data
		  y_t         - target label in L by 1 matrix
		  eig_ratio   - proportion/number of eigvalues used for test

		OUTPUT:
		  acc         - test accuracy on target domain
		  pre_labels  - predicted labels of target domain data
		"""

		vals = np.diag(A)
		vals_sqrt = np.zeros( (A.shape[0], ), dtype = float )
		ratio = []
		count = 0
		for i in range(0, len(vals)):
			if vals[i] < 0:
				break

			count  = count + vals[i]
			ratio.append(count)
			vals_sqrt[i] = 1 / np.sqrt(vals[i])

		A_sqrt = np.diag(vals_sqrt)
		ratio = [i/count for i in ratio ]

		if eig_ratio <= 1:
			element = [x for x in ratio if x > eig_ratio][0]
			idx = ratio.index(element)
			n_eigs = idx
		else:
			n_eigs = eig_ratio

		Zt = np.dot( np.dot( K_t.T, B[:, 0:n_eigs] ), A_sqrt[0:n_eigs, 0:n_eigs] )
		Zs = np.dot( np.dot( self.K_s_s_bar.T, B[:, 0:n_eigs] ), A_sqrt[0:n_eigs, 0:n_eigs] )
		neigh = KNeighborsClassifier(n_neighbors=1)
		neigh.fit(Zs, self.y_s)
		pre_labels = neigh.predict(Zt)
		acc = accuracy_score(y_t, pre_labels)
		return acc, pre_labels

	def learn(self):
		"""
		learn the optimal setting using the validatoin set
		"""

		if 'beta' not in self.params:
			self.params['beta'] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
			# self.params['beta'] = [0.1, 0.9]

		if 'alph' not in self.params:
			self.params['alph'] = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
			# self.params['alph'] = [1, 1e1]

		if 'gamm' not in self.params:
			self.params['gamm'] = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
			# self.params['gamm'] = [1, 1e1]

		if 'q_list' not in self.params:
			self.params['q_list'] = [2]

		if 'X_v' not in self.params and 'y_v' not in self.params:
			self.params['X_v'] = self.X_s
			self.params['y_v'] = self.y_s

		dist_s_v = cdist(self.X_s, self.params['X_v'])
		dist_s_v = dist_s_v**2
		sgm_v = compute_width(dist_s_v)
		K_s_v = np.exp( -dist_s_v / (2 * sgm_v * sgm_v) )
		n_v = self.params['X_v'].shape[0]
		H_v = np.eye(n_v, dtype = float) - np.ones((n_v, n_v), dtype = float) / n_v
		K_s_v_bar = np.dot( np.dot(self.H_s, K_s_v), H_v )

		print('Validating hyper-parameters ...')
		acc_mat = np.zeros( (len(self.params['q_list']), len(self.params['beta']), len(self.params['alph']), len(self.params['gamm'])) , dtype = float)
		for i in range(0, len(self.params['beta'])):
			for j in range(0, len(self.params['alph'])):
				for p in range(0, len(self.params['gamm'])):

					B, A = self._transformation(self.params['beta'][i], self.params['alph'][j], self.params['gamm'][p])
					for qidx in range(len(self.params['q_list'])):
						acc, _ = self._test(B, A, K_s_v_bar, self.params['y_v'], self.params['q_list'][qidx])
						acc_mat[qidx, i, j, p] = acc
						if self.params['verbose']:
							print('qidx: {}, i: {}, j: {}, p: {}, validation accuracy: {}'.format(qidx, i, j, p, acc) )

		idx = np.where(acc_mat == np.amax(acc_mat) )
		self.idx_best = idx
		print('Done!')

	def predict(self, X_t, y_t):
		"""
		predict the class labels of target domain instances

		INPUT:
		  X_t         - instances from target domain
		  y_t         - labels of instances in X_t

		OUTPUT:
		  acc         - test accuracy on target domain instances
		  pre_labels  - predicted labels of target domain instances
		"""
		idx_q, idx_i, idx_j, idx_p = self.idx_best[0], self.idx_best[1], self.idx_best[2], self.idx_best[3]

		# apply validated parameters on the test domain
		if self.params['verbose']:
			print('Classify the target domain instances ...')
		acc_final = 0
		labels_final = None
		for vidx in range(0, len(idx_q)):
			best_q = self.params['q_list'][ idx_q[vidx] ]
			best_beta = self.params['beta'][ idx_i[vidx] ]
			best_alph = self.params['alph'][ idx_j[vidx] ]
			best_gamm = self.params['gamm'][ idx_p[vidx] ]

			dist_s_t = cdist(self.X_s, X_t)
			dist_s_t = dist_s_t**2
			sgm_t = compute_width(dist_s_t)
			K_s_t = np.exp( -dist_s_t / (2 * sgm_t * sgm_t) )

			n_t = X_t.shape[0]
			H_t = np.eye(n_t, dtype = float) - np.ones((n_t, n_t), dtype = float) / n_t
			K_s_t_bar = np.dot( np.dot(self.H_s, K_s_t), H_t )

			B, A = self._transformation(best_beta, best_alph, best_gamm)
			acc_test, pre_labels = self._test(B, A, K_s_t_bar, y_t, best_q)

			if acc_test > acc_final:
				acc_final = acc_test
				labels_final = pre_labels

		print('\nThe final test accuracy: {}'.format(acc_final) )
		return acc_final, labels_final


def compute_width(dist_mat):
	n1, n2 = dist_mat.shape
	if n1 == n2:
		if np.allclose(dist_mat, dist_mat.T):
			id_tril = np.tril_indices(dist_mat.shape[0], -1)
			bandwidth = dist_mat[id_tril]
			return np.sqrt( 0.5 * np.median(bandwidth) )
		else:
			return np.sqrt( 0.5 * np.median(dist_mat) )
	else:
		return np.sqrt( 0.5 * np.median(dist_mat) )

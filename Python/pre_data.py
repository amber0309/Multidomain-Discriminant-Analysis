from __future__ import division
from scipy.io import loadmat
import numpy as np

def load_syn_data(fname = 'data2'):

	# load data
	tgt_dm = [4]
	val_dm = [2,3]
	src_dm = [0,1]
	data_dict = loadmat( './data/' + fname + '.mat' )
	data = data_dict['XY_cell']

	# test data
	X_t = data[0, tgt_dm[0]][:,0:2]
	Y_t = data[0, tgt_dm[0]][:,2]-1

	# training data
	X_s_list = []
	Y_s_list = []
	for s in range(0, len(src_dm)):
		cu_dm = src_dm[s]
		X_s_list.append( data[0, cu_dm][:,0:2] )
		Y_s_list.append( data[0, cu_dm][:,2]-1 )

	# validation data
	X_v = np.concatenate( (data[0, val_dm[0]][:,0:2],data[0, val_dm[1]][:,0:2]), axis=0)
	Y_v = np.concatenate( (data[0, val_dm[0]][:,2],data[0, val_dm[1]][:,2]), axis=0)-1
	X_s = np.concatenate(X_s_list)
	y_s = np.concatenate(Y_s_list)
	return X_s_list, Y_s_list, X_v, Y_v, X_t, Y_t

if __name__ == '__main__':
	load_syn_data()
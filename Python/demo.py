from pre_data import load_syn_data
from MDA import MDA

def demo():
	X_s_list, y_s_list, X_v, y_v, X_t, y_t = load_syn_data()
	params = {'X_v': X_v, 'y_v': y_v, 'verbose': True}
	mdl = MDA(X_s_list, y_s_list, params)
	mdl.learn()
	mdl.predict(X_t, y_t)

if __name__ == '__main__':
	demo()
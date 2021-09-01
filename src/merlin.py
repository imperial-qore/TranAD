# Replicated from the following paper:
# Nakamura, T., Imamura, M., Mercer, R. and Keogh, E., 2020, November. 
# MERLIN: Parameter-Free Discovery of Arbitrary Length Anomalies in Massive 
# Time Series Archives. In 2020 IEEE International Conference on Data Mining (ICDM) 
# (pp. 1190-1195). IEEE.

import numpy as np
from pprint import pprint
from time import time
from src.utils import *
from src.constants import *
from src.diagnosis import *
from src.pot import *
maxint = 200000

# z-normalized euclidean distance
def dist(t, q):
	m = q.shape[0]
	# t, q = t.reshape(-1), q.reshape(-1)
	# znorm2 = 2 * m * (1 - (np.dot(q, t) - m * np.mean(q) * np.mean(t)) / (m * np.std(q) * np.std(t)))
	znorm2 = np.mean((t - q) ** 2)
	return np.sqrt(znorm2)

# get L length subsequence from t starting at i
def getsub(t, L, i):
	return t[i:i+L]

# Candidate Selection Algorithm
def csa(t, L, r):
	C = []
	for i in range(1, t.shape[0] - L + 1):
		iscandidate = True
		for j in C:
			if i != j:
				if dist(getsub(t, L, i), getsub(t, L, j)) < r:
					C.remove(j)
					iscandidate = False
		if iscandidate and i not in C:
			C.append(i)
	if C:
		return C
	else:
		return []

# Checking function
def check(t, pred):
	labels = [];
	for i in range(t.shape[1]):
		new = np.convolve(t[:, i], np.ones(cvp)/cvp, mode='same')
		scores = np.abs(new - t[:,i])
		labels.append((scores > np.percentile(scores, percentile_merlin)) + 0)
	labels = np.array(labels).transpose()
	return (np.sum(labels, axis=1) >= 1) + 0, labels

# Discords Refinement Algorithm
def drag(C, t, L, r):
	D = [];
	if not C: return []
	for i in range(1, t.shape[0] - L + 1):
		isdiscord = True 
		dj = maxint
		for j in C:
			if i != j:
				d = dist(getsub(t, L, i), getsub(t, L, j))
				if d < r:
					C.remove(j)
					isdiscord = False
				else:
					dj = min(dj, d)
		if isdiscord:
			D.append((i, L, dj))
	return D

# MERLIN
def merlin(t, minL, maxL):
	r = 2 * np.sqrt(minL)
	dminL = - maxint; DFinal = []
	while dminL < 0:
		C = csa(t, minL, r)
		D = drag(C, t, minL, r)
		r = r / 2
		if D: break
	rstart = r
	distances = [-maxint] * 4
	print('phase 1')
	for i in range(minL, min(minL+4, maxL)):
		di = distances[i - minL]
		dim1 = rstart if i == minL else distances[i - minL - 1]
		r = 0.99 * dim1
		while di < 0:
			C = csa(t, i, r)
			D = drag(C, t, i, r)
			if D: 
				di = np.max([p[2] for p in D])
				distances[i - minL] = di
				DFinal += D
			r = r * 0.99
		print(i, r)
	print('phase 2')
	for i in range(minL + 4, maxL + 1):
		M = np.mean(distances)
		S = np.std(distances) + 1e-2
		r = M - 2 * S
		di = - maxint
		for _ in range(1000):
			C = csa(t, i, r)
			D = drag(C, t, i, r)
			if D: 
				di = np.max([p[2] for p in D])
				DFinal += D
				if di > 0:	break
			r = r - S
	vals = []
	for p in DFinal: 
		if p[2] != maxint: vals.append(p[2])
	dmin = np.argmax(vals)
	return DFinal[dmin], DFinal

def get_result(pred, labels):
	p_t = calc_point2point(pred, labels)
	result = {
        'f1': p_t[0],
        'precision': p_t[1],
        'recall': p_t[2],
        'TP': p_t[3],
        'TN': p_t[4],
        'FP': p_t[5],
        'FN': p_t[6],
        'ROC/AUC': p_t[7],
    }
	return result

def run_merlin(test, labels, dset):
	t = next(iter(test)).detach().numpy(); labelsAll = labels
	labels = (np.sum(labels, axis=1) >= 1) + 0
	lsum = np.sum(labels)
	start = time()
	pred = np.zeros_like(labels)
	d, _ = merlin(t, 60, 62) #
	print('Result:', d) #
	pred[d[0]:d[0]+d[1]] = 1; #
	pred, predAll = check(t, pred)
	print(t.shape, pred.shape, labels.shape)
	result = get_result(pred, labels)
	if dset in ['SMD', 'MSDS']:
		result.update(hit_att(predAll, labelsAll))
		result.update(ndcg(predAll, labelsAll))
	pprint(result); 
	print(color.BOLD+'Training time: '+"{:10.4f}".format(time()-start)+' s'+color.ENDC)
	exit()

if __name__ == '__main__':
	# simple test case (comment line 2 and run 'python3 src/merlin.py')
	a = np.random.normal(size=(100, 1))
	a[10:13][:] = 100
	d, D = merlin(a, 1, 10)		
	print(D); print(d)

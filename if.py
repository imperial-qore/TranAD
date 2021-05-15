from sklearn.ensemble import IsolationForest
from main import *
from tqdm import trange

if __name__ == '__main__':
	train_loader, test_loader, labels = load_dataset(args.dataset)

	## Prepare data
	trainD, testD = next(iter(train_loader)), next(iter(test_loader))

	### Training and Testing phase
	clf = IsolationForest(random_state=0, n_estimators=10, max_features=1.0, bootstrap=False)
	pred = []
	for i in trange(trainD.shape[1]):
		td = trainD[:, i].reshape(-1, 1)
		c = clf.fit(td.tolist())
		p = c.predict(testD[:, i].reshape(-1, 1).tolist())
		p = (p + 1) / 2
		pred.append(p)
	pred = np.array(pred).transpose()
	print(pred.shape)

	### Scores
	pred = (np.sum(pred, axis=1) >= 1) + 0
	labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
	print(pred, labelsFinal)
	p_t = calc_point2point(pred, labelsFinal)
	result = {
        'f1': p_t[0],
        'precision': p_t[1],
        'recall': p_t[2],
        'TP': p_t[3],
        'TN': p_t[4],
        'FP': p_t[5],
        'FN': p_t[6],
    }
	result.update(hit_att(pred, labelsFinal))
	result.update(ndcg(pred, labelsFinal))
	pprint(result)
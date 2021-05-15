from sklearn.ensemble import IsolationForest
from main import *
from tqdm import trange

rng = np.random.RandomState(42)

if __name__ == '__main__':
	train_loader, test_loader, labels = load_dataset(args.dataset)

	## Prepare data
	trainD, testD = next(iter(train_loader)), next(iter(test_loader))

	### Training and Testing phase
	clf = IsolationForest(random_state=rng, n_estimators=1000, max_features=1.0, bootstrap=False)
	# tpred, pred = [], []
	# for i in trange(trainD.shape[1]):
	# 	td = trainD[:, i].reshape(-1, 1)
	# 	c = clf.fit(td.tolist())
	# 	tp = c.predict(trainD[:, i].reshape(-1, 1).tolist())
	# 	p = c.predict(testD[:, i].reshape(-1, 1).tolist())
	# 	p = (-p + 1) / 2; tp = (-tp + 1) / 2
	# 	pred.append(p); tpred.append(tp)
	# pred = np.array(pred).transpose(); tpred = np.array(tpred).transpose()
	# pred, tpred = np.mean((-pred + 1) / 2, axis=1), np.mean((-tpred + 1) / 2, axis=1)
	c = clf.fit(trainD.tolist())
	pred, tpred = c.predict(testD.tolist()), c.predict(trainD.tolist())
	pred = (-pred + 1) / 2; tpred = (-tpred + 1) / 2

	### Scores
	labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
	print(pred.shape, tpred.shape, labelsFinal.shape)
	result, _ = pot_eval(tpred, pred, labelsFinal)
	# result.update(hit_att(pred, labelsFinal))
	# result.update(ndcg(pred, labelsFinal))
	pprint(result)
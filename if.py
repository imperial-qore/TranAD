from sklearn.ensemble import IsolationForest
from main import *


if __name__ == '__main__':
	train_loader, test_loader, labels = load_dataset(args.dataset)

	## Prepare data
	trainD, testD = next(iter(train_loader)), next(iter(test_loader))

	### Training phase
	clf = IsolationForest(random_state=0, n_estimators=100, max_features=1.0, bootstrap=True, verbose=True).fit(trainD.tolist())

	### Testing phase
	pred = clf.predict(testD.tolist())
	pred = (pred + 1) / 2

	### Scores
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
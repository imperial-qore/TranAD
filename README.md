[![License](https://img.shields.io/badge/License-BSD%203--Clause-red.svg)](https://github.com/imperial-qore/TranAD/blob/master/LICENSE)
![Python 3.7, 3.8](https://img.shields.io/badge/python-3.7%20%7C%203.8-blue.svg)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fimperial-qore%2FTranAD&count_bg=%23FFC401&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

# AnomalyDetection
Anomaly Detection using Transformers, self-conditioning and adversarial training.

## Novelty

1. transformer :
	- parallelized quick training
	- self-attention for local contextual trend

2. self-conditioning :
	- multi-modal feature extraction
	- prevent mode collapse
	- know focus points (focus score) (ref USAD sec 3.2 para 2)

3. adversarial taining :
	- amplify reconstruction error and gain stability

4. model-agnostic meta learning :
	- learning with limited data



## Figures and Comparisons

- Model diagram (Figure) :
	- Transformer model
	- Two-phase inference 
	- Adversarial training
	- Self-Conditioning focus score

- Visualization of attention score (Plot) :
	- Truncated (say 5 dimension time-series data) with colormap of attention scores for each dimension (like github)

- Visualization of encodings (Plot) : (like in AAAI 2021 paper)
	- Plot of ground truth, prediction, label, anomaly score with tSNE
	- tSNE plot of embeddings on SMD dataset for different time-series dimensions.

- Dataset Statistics (Table) :
	- Dimension
	- Size (training and testing)
	- Anomaly rate

- Detection (Table) [1, 2, 3] :
	- complete datasets - F1, Precision, Recall
	- partial datasets  - F1, Precision, Recall (performance with limited training data)

- Training Time (Table) [1] :
	- complete and partial datasets 

- Evaluation with different delays (Plot) [4] : (like in ICDM 2020 paper)
	- performance with increasing delta value

- Robustness to noise (Plot) [4] : (like in MSCRED paper)
	- performance with increasing noise value

- Diagnosis (Table) [2] :
	- complete datasets - NDCG@5, HitRate@100%, HitRate@150%
	- partial datasets  - NDCG@5, HitRate@100%, HitRate@150%

- Scalability (Plot) [1] :
	- F1 score and training time with increasing dataset size (20%, 40%, 60%, 80%, 100%)

- Ablation (Table) [1, 2, 3, 4] :
	- Detection, delay and diagnosis performance w/o MAML/self-conditioning/adversarial-training

- Sensitivity Analysis (Plots) : (like in USAD paper)
	- P, R, F1, F1\*, training time with learning rate, number of layers, window size, % anomalies, 
	- F1 and training time with window size

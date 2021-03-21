# AnomalyDetection
Anomaly Detection using Transformers, self-conditioning and adversarial training.

## NOVELTY

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



## FIGURES AND COMPARISONS


Model diagram (Figure) :
	- Transformer model
	- Two-phase inference 
	- Adversarial training
	- Self-Conditioning focus score

Visualization of attention score (Plot) :
	- Truncated (say 5 dimension time-series data) with colormap of attention scores for each dimension (like github)

Dataset Statistics (Table) :
	- Dimension
	- Size (training and testing)
	- Anomaly rate

Detection (Table) [1, 2, 3] :
	- complete datasets - F1, Precision, Recall
	- partial datasets  - F1, Precision, Recall (performance with limited training data)

Training Time (Table) [1] :
	- complete and partial datasets 

Evaluation with different delays (Plot) [4] : (like in ICDM 2020 paper)
	- performance with increasing delta value

Robustness to noise (Plot) [4] : (like in MSCRED paper)
	- performance with increasing noise value

Diagnosis (Table) [2] :
	- complete datasets - NDCG@5, HitRate@100%, HitRate@150%
	- partial datasets  - NDCG@5, HitRate@100%, HitRate@150%

Ablation (Table) [1, 2, 3, 4] :
	- Detection, delay and diagnosis performance w/o MAML/self-conditioning/adversarial-training
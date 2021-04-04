from src.parser import *
from src.folderconstants import *

# Threshold parameters
lm_d = {
		'SMD': (0.99995, 1.04), 
		'synthetic': (0.999, 1),
		'SWaT': (0.993, 1),
		'SMAP': (0.97, 1),
		'MSL': (0.99905, 1),
		'WADI': (0.99, 1),
		'MSDS': (0.91, 1)
	}
lm = lm_d[args.dataset]

# Hyperparameters
lr_d = {
		'SMD': 0.0001, 
		'synthetic': 0.0001, 
		'SWaT': 0.003, 
		'SMAP': 0.0001, 
		'MSL': 0.0001, 
		'WADI': 0.0001, 
		'MSDS': 0.001, 
	}
lr = lr_d[args.dataset]

# Debugging

preds = []
debug = 9
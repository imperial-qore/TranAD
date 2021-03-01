import matplotlib.pyplot as plt
import statistics
import os
import numpy as np

plt.style.use(['science', 'ieee'])
plt.rcParams["text.usetex"] = True
plt.rcParams['figure.figsize'] = 6, 2

os.makedirs('plots', exist_ok=True)

def smooth(y, box_pts=1):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def plotter(name, y_true, y_pred, ascore, labels):
	os.makedirs(os.path.join('plots', name), exist_ok=True)
	for dim in range(y_true.shape[1]):
		y_t, y_p, l, a_s = y_true[:, dim], y_pred[:, dim], labels[:, dim], ascore[:, dim]
		fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
		ax1.plot(smooth(y_t), label='True')
		ax1.plot(smooth(y_p), label='Predicted')
		ax1.plot(l, '--', alpha=0.5)
		ax1.legend()
		ax2.plot(smooth(a_s))
		fig.savefig(f'plots/{name}/{dim}.pdf')

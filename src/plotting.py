import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import statistics
import os, torch
import numpy as np
import scienceplots

plt.style.use(['science', 'ieee'])
plt.rcParams["text.usetex"] = False
plt.rcParams['figure.figsize'] = 6, 2

os.makedirs('plots', exist_ok=True)

def smooth(y, box_pts=1):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def plot_curve(y_t, y_p, l, a_s, p, pdf, title, final=False, first=False, thresh=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.set_ylabel('Value')
    ax1.set_title(title)
    if not final:
        # if dim == 0: np.save(f'true{dim}.npy', y_t); np.save(f'pred{dim}.npy', y_p); np.save(f'ascore{dim}.npy', a_s)
        ax1.plot(smooth(y_t), linewidth=0.2, label='True')
        ax1.plot(smooth(y_p), '-', alpha=0.6, linewidth=0.3, label='Predicted')
    ax3 = ax1.twinx()
    ax3.plot(l, '--', linewidth=0.3, alpha=0.5)
    ax3.fill_between(np.arange(l.shape[0]), l, color='blue', alpha=0.3)
    ax4 = ax1.twinx()
    ax4.plot(p, '--', linewidth=0.3, alpha=0.5)
    ax4.fill_between(np.arange(p.shape[0]), p, color='red', alpha=0.3)
    if first: ax1.legend(ncol=2, bbox_to_anchor=(0.6, 1.02))
    ax2.plot(smooth(a_s), linewidth=0.2, color='g')
    ax2.axhline(y=thresh, color='r', linestyle='--', label='Threshold')
    ax2.set_xlabel('Timestamp')
    ax2.set_ylabel('Anomaly Score')
    pdf.savefig(fig)
    plt.close()

def plotter(name, y_true, y_pred, ascore, labels, preds, ascore_final, preds_final, thresh, thresh_final, is_veremi=False):
    # if 'TranAD' in name or 'Alladi' in name: y_true = torch.roll(y_true, 1, 0)
	os.makedirs(os.path.join('plots', name), exist_ok=True)
	pdf = PdfPages(f'plots/{name}/output.pdf')
	for dim in range(y_true.shape[2]):
		curr_thresh = thresh[dim]
	#	labelsF = labels[:200, 0] if is_veremi else labels[:200, dim]
	#	y_t, y_p, l, a_s, p = y_true[-1, :200, dim], y_pred[:200, dim], np.where(labelsF[:200] > 0, 1, 0), ascore[:200, dim], preds[:200, dim]
	#	title = f'Dimension = {dim}'
	#	plot_curve(y_t, y_p, l, a_s, p, pdf, title, first=dim == 0, thresh=curr_thresh)
	#a_s, p = ascore_final[:200], preds_final[:200]
		labelsF = labels[:, 0] if is_veremi else labels[:, dim]
		y_t, y_p, l, a_s, p = y_true[-1, :, dim], y_pred[:, dim], np.where(labelsF[:] > 0, 1, 0), ascore[:, dim], preds[:, dim]
		title = f'Dimension = {dim}'
		plot_curve(y_t, y_p, l, a_s, p, pdf, title, first=dim == 0, thresh=curr_thresh)
	a_s, p = ascore_final[:], preds_final[:]
	plot_curve(None, None, l, a_s, p, pdf, title="All dimensions", final=True, thresh=thresh_final)
	pdf.close()
	

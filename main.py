import gc
import pickle
import os
import pandas as pd
import json
from tqdm import tqdm
from src.models import *
from src.constants import *
from src.plotting import *
from src.pot import *
from src.utils import *
from src.diagnosis import *
from src.merlin import *
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
from time import time
from pprint import pprint
import h5py
# from beepy import beep


class HDF5Dataset(Dataset):
    def __init__(self, h5_data, chunk_size=1000, device='cpu', less=False):
        self.h5_data = h5_data
        if less:
            self.h5_data = cut_array_window_first(0.2, self.h5_data)
        self.device = device
        self.chunk_size = chunk_size
        self.chunk_start = -1
        self.chunk_end = 0
        self.chunk(0)
        
    def chunk(self, idx):
        if  idx < self.chunk_start or idx >= self.chunk_end:
            self.chunk_start = (idx // self.chunk_size) * self.chunk_size
            self.chunk_end = self.chunk_start + self.chunk_size
            chunk = self.h5_data[:, self.chunk_start:self.chunk_end]
            if hasattr(self, '_chunk'):
                del self._chunk
            self._chunk = torch.from_numpy(chunk).float().to(self.device)
        bounded_idx = idx - self.chunk_start
        return self._chunk, bounded_idx

    def __len__(self):
        return self.h5_data.shape[1]

    def __getitem__(self, idx):
        chunk, bounded_idx = self.chunk(idx)
        data = chunk[:, bounded_idx]
        return data, data

def convert_to_windows(data, model, training=True):
	windows = []
	w_size = model.n_window
	slide = 1
	start = 0
	if training:
		if hasattr(model, 'n_window_slide'):
			slide = model.n_window_slide
		if hasattr(model, 'n_window_start'):
			start = model.n_window_start

	for i in range(start, len(data), slide): 
		if i >= w_size:
			w = data[i-w_size:i]
		else:
			w = torch.cat([data[0].repeat(w_size-i, 1), data[0:i]])
		windows.append(w if 'TranAD' in args.model or 'Attention' in args.model or 'Alladi' in args.model else w.view(-1))
	out = torch.stack(windows)
	return out.permute(1, 0, 2)

def load_dataset(dataset, device):
	folder = os.path.join(output_folder, dataset)
	if not os.path.exists(folder):
		raise Exception('Processed Data not found.')
	loader = []

	if 'VeReMiH5' in dataset:
		f = h5py.File(os.path.join(folder, 'veremi.h5'))
		test = f['test']
		labels = f['test_labels'][:]
		if '95' in dataset:
			train = f['train_95_genuine']
		elif '90' in dataset:
			train = f['train_90_genuine']
		else:
			train = f['train_full_genuine']
		return train, test, labels

	for file in ['train', 'test', 'labels']:
		if dataset == 'SMD': file = 'machine-1-1_' + file
		if dataset == 'SMAP': file = 'P-1_' + file
		if dataset == 'MSL': file = 'C-1_' + file
		if dataset == 'UCR': file = '136_' + file
		if dataset == 'NAB': file = 'ec2_request_latency_system_failure_' + file
		loader.append(np.load(os.path.join(folder, f'{file}.npy')))
	# loader = [i[:, debug:debug+1] for i in loader]
	if args.less: loader[0] = cut_array(0.2, loader[0])
	train = torch.tensor(loader[0], device=device, dtype=torch.float32)
	test = torch.tensor(loader[1], device=device, dtype=torch.float32)
	labels = loader[2]
	return train, test, labels

def save_results(epoch, df, result):
	folder = f'result_metrics/{args.model}_{args.dataset}/'
	os.makedirs(folder, exist_ok=True)
	file_path = f'{folder}/results-e{epoch}.txt'
	with open(file_path, 'w+') as fp:
		fp.writelines(str(df))
		fp.write('\n')
		pprint(result, stream=fp)

def save_model(model, optimizer, scheduler, epoch, accuracy_list):
	folder = f'checkpoints/{args.model}_{args.dataset}/'
	os.makedirs(folder, exist_ok=True)
	file_path = f'{folder}/model-e{epoch}.ckpt'
	torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list}, file_path)

def load_model(modelname, dims, device=None, parallel=False):
	import src.models
	model_class = getattr(src.models, modelname)
	model = model_class(dims) #  .double()
	model.to(device)
	optimizer = torch.optim.AdamW(model.parameters() , lr=model.lr, weight_decay=1e-5)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
	folder = f'checkpoints/{args.model}_{args.dataset}/'
	files = os.listdir(folder) if os.path.exists(folder) else None
	if files:
		files.sort(key=lambda x: os.path.getmtime(folder + x))
		most_recent_checkpoint = files[-1]
		fname = folder + most_recent_checkpoint
	else:
		fname = folder + 'model.ckpt'
	if parallel:
		p_model = nn.DataParallel(model)
		p_model.name = model.name
		p_model.batch = model.batch
		model = p_model
	if os.path.exists(fname) and (not args.retrain or args.test):
		print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
		checkpoint = torch.load(fname, map_location=device)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		epoch = checkpoint['epoch']
		accuracy_list = checkpoint['accuracy_list']
	else:
		print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
		epoch = -1; accuracy_list = []
	return model, optimizer, scheduler, epoch, accuracy_list

def backprop(epoch, model, data, optimizer, scheduler, device, training = True, training_data = False):
	if training:
		model.train(True)
	l = nn.MSELoss(reduction = 'mean' if training else 'none')
	feats = data.shape[-1]
	if 'DAGMM' in model.name:
		l = nn.MSELoss(reduction = 'none')
		compute = ComputeLoss(model, 0.1, 0.005, 'cpu', model.n_gmm)
		n = epoch + 1; w_size = model.n_window
		l1s = []; l2s = []
		if training:
			for d in data:
				_, x_hat, z, gamma = model(d)
				l1, l2 = l(x_hat, d), l(gamma, d)
				l1s.append(torch.mean(l1).item()); l2s.append(torch.mean(l2).item())
				loss = torch.mean(l1) + torch.mean(l2)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			scheduler.step()
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
			return np.mean(l1s)+np.mean(l2s), optimizer.param_groups[0]['lr']
		else:
			ae1s = []
			for d in data: 
				_, x_hat, _, _ = model(d)
				ae1s.append(x_hat)
			ae1s = torch.stack(ae1s)
			y_pred = ae1s[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			loss = l(ae1s, data)[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			return loss.detach().numpy(), y_pred.detach().numpy()
	if 'Attention' in model.name:
		l = nn.MSELoss(reduction = 'none')
		n = epoch + 1; w_size = model.n_window
		l1s = []; res = []
		if training:
			for d in data:
				ae, ats = model(d)
				# res.append(torch.mean(ats, axis=0).view(-1))
				l1 = l(ae, d)
				l1s.append(torch.mean(l1).item())
				loss = torch.mean(l1)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			# res = torch.stack(res); np.save('ascores.npy', res.detach().numpy())
			scheduler.step()
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
			return np.mean(l1s), optimizer.param_groups[0]['lr']
		else:
			ae1s, y_pred = [], []
			for d in data: 
				ae1 = model(d)
				y_pred.append(ae1[-1])
				ae1s.append(ae1)
			ae1s, y_pred = torch.stack(ae1s), torch.stack(y_pred)
			loss = torch.mean(l(ae1s, data), axis=1)
			return loss.detach().numpy(), y_pred.detach().numpy()
	elif 'OmniAnomaly' in model.name:
		if training:
			mses, klds = [], []
			for i, d in enumerate(data):
				y_pred, mu, logvar, hidden = model(d, hidden if i else None)
				MSE = l(y_pred, d)
				KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)
				loss = MSE + model.beta * KLD
				mses.append(torch.mean(MSE).item()); klds.append(model.beta * torch.mean(KLD).item())
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tKLD = {np.mean(klds)}')
			scheduler.step()
			return loss.item(), optimizer.param_groups[0]['lr']
		else:
			y_preds = []
			for i, d in enumerate(data):
				y_pred, _, _, hidden = model(d, hidden if i else None)
				y_preds.append(y_pred)
			y_pred = torch.stack(y_preds)
			MSE = l(y_pred, data)
			return MSE.detach().numpy(), y_pred.detach().numpy()
	elif 'USAD' in model.name:
		l = nn.MSELoss(reduction = 'none')
		n = epoch + 1; w_size = model.n_window
		l1s, l2s = [], []
		if training:
			for d in data:
				ae1s, ae2s, ae2ae1s = model(d)
				l1 = (1 / n) * l(ae1s, d) + (1 - 1/n) * l(ae2ae1s, d)
				l2 = (1 / n) * l(ae2s, d) - (1 - 1/n) * l(ae2ae1s, d)
				l1s.append(torch.mean(l1).item()); l2s.append(torch.mean(l2).item())
				loss = torch.mean(l1 + l2)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			scheduler.step()
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
			return np.mean(l1s)+np.mean(l2s), optimizer.param_groups[0]['lr']
		else:
			ae1s, ae2s, ae2ae1s = [], [], []
			for d in data: 
				ae1, ae2, ae2ae1 = model(d)
				ae1s.append(ae1); ae2s.append(ae2); ae2ae1s.append(ae2ae1)
			ae1s, ae2s, ae2ae1s = torch.stack(ae1s), torch.stack(ae2s), torch.stack(ae2ae1s)
			y_pred = ae1s[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			loss = 0.1 * l(ae1s, data) + 0.9 * l(ae2ae1s, data)
			loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			return loss.detach().numpy(), y_pred.detach().numpy()
	elif model.name in ['GDN', 'MTAD_GAT', 'MSCRED', 'CAE_M']:
		l = nn.MSELoss(reduction = 'none')
		n = epoch + 1; w_size = model.n_window
		l1s = []
		if training:
			for i, d in enumerate(data):
				if 'MTAD_GAT' in model.name: 
					x, h = model(d, h if i else None)
				else:
					x = model(d)
				loss = torch.mean(l(x, d))
				l1s.append(torch.mean(loss).item())
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(l1s)}')
			return np.mean(l1s), optimizer.param_groups[0]['lr']
		else:
			xs = []
			for d in data: 
				if 'MTAD_GAT' in model.name: 
					x, h = model(d, None)
				else:
					x = model(d)
				xs.append(x)
			xs = torch.stack(xs)
			y_pred = xs[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			loss = l(xs, data)
			loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			return loss.detach().numpy(), y_pred.detach().numpy()
	elif 'GAN' in model.name:
		l = nn.MSELoss(reduction = 'none')
		bcel = nn.BCELoss(reduction = 'mean')
		msel = nn.MSELoss(reduction = 'mean')
		real_label, fake_label = torch.tensor([0.9]), torch.tensor([0.1]) # label smoothing
		real_label, fake_label = real_label.type(torch.DoubleTensor), fake_label.type(torch.DoubleTensor)
		n = epoch + 1; w_size = model.n_window
		mses, gls, dls = [], [], []
		if training:
			for d in data:
				# training discriminator
				model.discriminator.zero_grad()
				_, real, fake = model(d)
				dl = bcel(real, real_label) + bcel(fake, fake_label)
				dl.backward()
				model.generator.zero_grad()
				optimizer.step()
				# training generator
				z, _, fake = model(d)
				mse = msel(z, d) 
				gl = bcel(fake, real_label)
				tl = gl + mse
				tl.backward()
				model.discriminator.zero_grad()
				optimizer.step()
				mses.append(mse.item()); gls.append(gl.item()); dls.append(dl.item())
				# tqdm.write(f'Epoch {epoch},\tMSE = {mse},\tG = {gl},\tD = {dl}')
			tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tG = {np.mean(gls)},\tD = {np.mean(dls)}')
			return np.mean(gls)+np.mean(dls), optimizer.param_groups[0]['lr']
		else:
			outputs = []
			for d in data: 
				z, _, _ = model(d)
				outputs.append(z)
			outputs = torch.stack(outputs)
			y_pred = outputs[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			loss = l(outputs, data)
			loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			return loss.detach().numpy(), y_pred.detach().numpy()
	elif 'TranAD' in model.name:
		l = nn.MSELoss(reduction = 'none')
		bs = model.batch if training else 5000
		if 'VeReMiH5' in args.dataset:
			dataset = HDF5Dataset(data, chunk_size=bs*100, device=device, less=args.less and training_data)
		else:
			data = data.permute(1, 0, 2)
			dataset = TensorDataset(data, data)
		dataloader = DataLoader(dataset, batch_size=bs)
		n = epoch + 1
		l1s, l2s = [], []
		if training:
			for d, _ in tqdm(dataloader):
				d = d.to(device)
                # pack padded sequence
				local_bs = d.shape[0]
				window = d.permute(1, 0, 2)
				elem = window[-1, :, :].view(1, local_bs, feats)
				z = model(window, elem)
				# old loss
				# l1 = l(z, elem) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1/n) * l(z[1], elem)
				l1 = l(z, elem) if not isinstance(z, tuple) else (0.9 ** n) * l(z[0], elem) + (1 - 0.9 ** n) * l(z[1], elem)
				l1 += 0 if not isinstance(z, tuple) or len(z) < 3 else  (0.9 ** n) * l(z[2], elem) - (1 - 0.9 ** n) * l(z[1], elem)
				if isinstance(z, tuple): z = z[1]
				l1s.append(torch.mean(l1).item())
				loss = torch.mean(l1)
				optimizer.zero_grad()
				loss.backward(retain_graph=True)
				optimizer.step()
			scheduler.step()
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
			return np.mean(l1s), optimizer.param_groups[0]['lr']
		else:
			losses = []
			zs = []
			for d, _ in tqdm(dataloader):
				d = d.to(device)
				local_bs = d.shape[0]
				window = d.permute(1, 0, 2)
				elem = window[-1, :, :].view(1, local_bs, feats)
				z = model(window, elem)
				if isinstance(z, tuple): z = z[1]
				loss = l(z, elem)[0]
				zs.append(z.cpu().detach())
				losses.append(loss.cpu().detach())	
			loss = torch.cat(losses, 0)
			z = torch.cat(zs, 1)
			return loss.detach().cpu().numpy(), z.detach().cpu().numpy()[0]
	elif 'Alladi' in model.name:
		l = nn.MSELoss(reduction = 'none')
		bs = model.batch if training else 5000
		if 'VeReMiH5' in args.dataset:
			dataset = HDF5Dataset(data, chunk_size=bs*100, device=device, less=args.less and training)
		else:
			data = data.permute(1, 0, 2)
			dataset = TensorDataset(data, data)
		dataloader = DataLoader(dataset, batch_size = bs)
		n = epoch + 1
		l1s, l2s = [], []
		if training:
			for d, _ in tqdm(dataloader):
				d = d.to(device)
				local_bs = d.shape[0]
				window = d.permute(1, 0, 2)
				elem = window[-1, :, :].view(1, local_bs, feats)
				z = model(window)
				l1 = l(z, elem)
				l1s.append(torch.mean(l1).item())
				loss = torch.mean(l1)
				optimizer.zero_grad()
				loss.backward(retain_graph=True)
				optimizer.step()
			scheduler.step()
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
			return np.mean(l1s), optimizer.param_groups[0]['lr']
		else:
			losses = []
			zs = []
			for d, _ in tqdm(dataloader):
				d = d.to(device)
				local_bs = d.shape[0]
				window = d.permute(1, 0, 2)
				elem = window[-1, :, :].view(1, local_bs, feats)
				z = model(window)
				loss = l(z, elem)[0]
				zs.append(z.cpu().detach())
				losses.append(loss.cpu().detach())
			loss = torch.cat(losses, 0)
			z = torch.cat(zs, 1)
			return loss.detach().cpu().numpy(), z.detach().cpu().numpy()[0]
	else:
		y_pred = model(data)
		loss = l(y_pred, data)
		if training:
			tqdm.write(f'Epoch {epoch},\tMSE = {loss}')
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			scheduler.step()
			return loss.item(), optimizer.param_groups[0]['lr']
		else:
			return loss.detach().cpu().numpy(), y_pred.detach().cpu().numpy()

if __name__ == '__main__':
	exec_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
	cpu = torch.device("cpu")
	train, test, labels = load_dataset(args.dataset, cpu)

	if args.model in ['MERLIN']:
		eval(f'run_{args.model.lower()}(test_loader, labels, args.dataset)')
	dims = test.shape[-1]
	model, optimizer, scheduler, epoch, accuracy_list = load_model(args.model, dims, device=exec_device, parallel=args.parallel)

	## Prepare data
	if 'VeReMi' not in args.dataset and (model.name in ['Attention', 'DAGMM', 'USAD', 'MSCRED', 'CAE_M', 'GDN', 'MTAD_GAT', 'MAD_GAN', 'AlladiCNNLSTM'] or 'TranAD' in model.name):
		train = convert_to_windows(train, model)

	n_trainings = args.n_train if not args.test else 1
	for n_training in range(n_trainings):
		### Training phase
		if not args.test:
			print(f'{color.HEADER}Training {args.model} on {args.dataset}{color.ENDC}')
			num_epochs = 5; e = epoch + 1; start = time()
			for e in tqdm(list(range(epoch+1, epoch+num_epochs+1))):
				lossT, lr = backprop(e, model, train, optimizer, scheduler, exec_device, training_data=True)
				accuracy_list.append((lossT, lr))
			print(color.BOLD+'Training time: '+"{:10.4f}".format(time()-start)+' s'+color.ENDC)
			save_model(model, optimizer, scheduler, e, accuracy_list)
			plot_accuracies(accuracy_list, f'{args.model}_{args.dataset}')
			del lossT
			del lr
			epoch += num_epochs

		if n_training == 0:
			if 'VeReMi' not in args.dataset and (model.name in ['Attention', 'DAGMM', 'USAD', 'MSCRED', 'CAE_M', 'GDN', 'MTAD_GAT', 'MAD_GAN', 'AlladiCNNLSTM'] or 'TranAD' in model.name): 
				test = convert_to_windows(test, model, training=False)

		### Testing phase
		torch.zero_grad = True
		model.eval()
		print(f'{color.HEADER}Testing {args.model} on {args.dataset}{color.ENDC}')
		loss, y_pred = backprop(0, model, test, optimizer, scheduler, exec_device, training=False, training_data=False)

		### Scores
		df = pd.DataFrame()
		lossT, _ = backprop(0, model, train, optimizer, scheduler, exec_device, training=False, training_data=True)

		preds = []
		threshs = []
		for i in tqdm(range(loss.shape[1])):
			lt, l, ls = lossT[:, i], loss[:, i], labels[:, 0] if 'VeReMi' in args.dataset else labels[:, i]
			result, pred = pot_eval(lt, l, ls)
			threshs.append(result['threshold'])
			preds.append(pred)
			# df = df.append(result, ignore_index=True)
			df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)

		# preds = np.concatenate([i.reshape(-1, 1) + 0 for i in preds], axis=1)
		# pd.DataFrame(preds, columns=[str(i) for i in range(10)]).to_csv('labels.csv')
		lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)
		if 'VeReMi' in args.dataset:
			if args.multilabel_test:
				labelsFinal = labels
			else:
				labelsFinal = labels[:, 0]
		else:
			labelsFinal = (np.sum(labels, axis=1) >= 1) + 0

		result, predsFinal = pot_eval(lossTfinal, lossFinal, labelsFinal, multi=args.multilabel_test)

		### Plot curves
		if args.plot or not args.test:
			preds = np.swapaxes(np.vstack(preds), 0, 1)
			plotter(f'{args.model}_{args.dataset}', test, y_pred, loss, labels, preds, lossFinal, predsFinal, thresh=threshs, thresh_final=result['threshold'], is_veremi='VeReMi' in args.dataset)

		# result.update(hit_att(loss, labels[n]))
		# result.update(ndcg(loss, labels[n]))
		print('ndcg')
		print(df)
		pprint(result)
		save_results(epoch, df, result)	
		# pprint(getresults2(df, result))
		# beep(4)

import pickle
import os
import pandas as pd
from src.parser import *
from src.models import *
from src.constants import *
from src.plotting import *
from src.pot import *
from src.utils import *
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from pprint import pprint

def convert_to_windows(data):
	windows = []
	for i, g in enumerate(data): 
		if i >= w_size: windows.append(data[i-w_size:i].view(-1))
		else: windows.append(torch.cat([data[0].repeat(w_size-i, 1), data[0:i]]).view(-1))
	return torch.stack(windows)

def load_dataset(dataset):
	folder = os.path.join(output_folder, dataset)
	if not os.path.exists(folder):
		raise Exception('Processed Data not found.')
	loader = []
	for file in ['train', 'test', 'labels']:
		if dataset == 'SMD': file = 'machine-1-1_' + file
		if dataset == 'SMAP': file = 'P-1_' + file
		if dataset == 'MSL': file = 'F-7_' + file
		loader.append(np.load(os.path.join(folder, f'{file}.npy')))
	# loader = [i[:, 1:2] for i in loader]
	train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
	test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])
	labels = loader[2]
	return train_loader, test_loader, labels

def save_model(model, optimizer, scheduler, epoch, accuracy_list):
	folder = f'checkpoints/{args.model}_{args.dataset}/'
	os.makedirs(folder, exist_ok=True)
	file_path = f'{folder}/model.ckpt'
	torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list}, file_path)

def load_model(modelname, dims):
	import src.models
	model_class = getattr(src.models, modelname)
	model = model_class(dims).double()
	optimizer = torch.optim.AdamW(model.parameters() , lr=model.lr, weight_decay=1e-5)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.8)
	fname = f'checkpoints/{args.model}_{args.dataset}/model.ckpt'
	if os.path.exists(fname):
		print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
		checkpoint = torch.load(fname)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		epoch = checkpoint['epoch']
		accuracy_list = checkpoint['accuracy_list']
	else:
		print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
		epoch = -1; accuracy_list = []
	return model, optimizer, scheduler, epoch, accuracy_list

def backprop(epoch, model, data, optimizer, scheduler, training = True):
	l = nn.MSELoss(reduction = 'mean' if training else 'none')
	if 'VAE' in model.name:
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
			print(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tKLD = {np.mean(klds)}')
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
		n = epoch + 1
		l1s, l2s = [], []
		if training:
			for d in data:
				ae1s, ae2s, ae2ae1s = model(d)
				l1 = (1 / n) * l(ae1s, d) + (1 - 1/n) * l(ae2ae1s, d)
				l2 = (1 / n) * l(ae2s, d) + (1 - 1/n) * l(ae2ae1s, d)
				l1s.append(torch.mean(l1).item()); l2s.append(torch.mean(l2).item())
				loss = torch.mean(l1 + l2)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			print(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
			return np.mean(l1s)+np.mean(l2s), optimizer.param_groups[0]['lr']
		else:
			feats = data.shape[1] // w_size
			ae1s, ae2s, ae2ae1s = [], [], []
			for d in data: 
				ae1, ae2, ae2ae1 = model(d)
				ae1s.append(ae1); ae2s.append(ae2); ae2ae1s.append(ae2ae1)
			ae1s, ae2s, ae2ae1s = torch.stack(ae1s), torch.stack(ae2s), torch.stack(ae2ae1s)
			y_pred = ae1s[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
			loss = torch.mean(0.1 * l(ae1s, data) + 0.9 * l(ae2ae1s, data), dim=1).view(-1,1).expand(-1, feats)
			return loss.detach().numpy(), y_pred.detach().numpy()
	else:
		y_pred = model(data)
		loss = l(y_pred, data)
		if training:
			print(f'Epoch {epoch},\tMSE = {MSE}')
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			scheduler.step()
			return loss.item(), optimizer.param_groups[0]['lr']
		else:
			return loss.detach().numpy(), y_pred.detach().numpy()

if __name__ == '__main__':
	train_loader, test_loader, labels = load_dataset(args.dataset)
	model, optimizer, scheduler, epoch, accuracy_list = load_model(args.model, labels.shape[1])

	## Prepare data
	trainD, testD = next(iter(train_loader)), next(iter(test_loader))
	trainO, testO = trainD, testD
	if 'USAD' in model.name: 
		trainD, testD = convert_to_windows(trainD), convert_to_windows(testD)

	### Training phase
	print(f'Training {args.model} on {args.dataset}')
	num_epochs = 5; e = epoch + 1
	for e in range(epoch+1, epoch+num_epochs+1):
		lossT, lr = backprop(e, model, trainD, optimizer, scheduler)
		accuracy_list.append((lossT, lr))
	save_model(model, optimizer, scheduler, e, accuracy_list)
	plot_accuracies(accuracy_list, f'{args.model}_{args.dataset}')

	### Testing phase
	torch.zero_grad = True
	model.eval()
	print(f'Testing {args.model} on {args.dataset}')
	loss, y_pred = backprop(0, model, testD, optimizer, scheduler, training=False)

	### Plot curves
	plotter(f'{args.model}_{args.dataset}', testO, y_pred, loss, labels)

	### Scores
	df = pd.DataFrame()
	lossT, _ = backprop(0, model, trainD, optimizer, scheduler, training=False)
	for i in range(loss.shape[1]):
		lt, l, ls = lossT[:, i], loss[:, i], labels[:, i]
		result = pot_eval(lt, l, ls, args.dataset)
		df = df.append(result, ignore_index=True)
	lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)
	labelsFinal = (np.sum(labels, axis=1) >= 1) + np.zeros(labels.shape[0])
	result = pot_eval(lossTfinal, lossFinal, labelsFinal, args.dataset)
	print(df)
	pprint(result)
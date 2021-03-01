import pickle
import os
import pandas as pd
from src.parser import *
from src.models import *
from src.constants import *
from src.plotting import *
from src.pot import *
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from pprint import pprint

def load_dataset(dataset):
	folder = os.path.join(output_folder, dataset)
	if not os.path.exists(folder):
		raise Exception('Processed Data not found.')
	loader = []
	for file in ['train', 'test', 'labels']:
		if dataset == 'SMD': file = 'machine-1-1_' + file
		with open(os.path.join(folder, f'{file}.pkl'), 'rb') as f:
			loader.append(pickle.load(f))
	loader = [i[:, 1:2] for i in loader]
	train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
	test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])
	labels = loader[2]
	return train_loader, test_loader, labels

def load_model(modelname, dims):
	import src.models
	model_class = getattr(src.models, modelname)
	model = model_class(dims).double()
	optimizer = torch.optim.Adam(model.parameters() , lr=model.lr, weight_decay=1e-5)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.5)
	return model, optimizer, scheduler

def backprop(model, data, optimizer, scheduler, training = True):
	l = nn.MSELoss(reduction = 'sum' if training else 'none')
	inp, y_true = data[:-1], data[1:]
	if 'VAE' in model.name:
		y_pred, mu, logvar = model(inp, training)
		MSE = l(y_pred, y_true)
		KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
		loss = MSE + model.beta * KLD
		if training:
			print(f'Epoch {epoch},\tMSE = {MSE},\tKLD = {model.beta * KLD}')
	else:
		y_pred = model(inp)
		MSE = l(y_pred, y_true)
		loss = MSE
		if training:
			print(f'Epoch {epoch},\tMSE = {MSE}')
	if not training:
		return MSE.detach().numpy(), y_pred.detach().numpy()
	else:
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		scheduler.step()
	return loss.item(), None

if __name__ == '__main__':
	train_loader, test_loader, labels = load_dataset(args.dataset)
	model, optimizer, scheduler = load_model(args.model, labels.shape[1])

	### Training phase
	print(f'Training {args.model} on {args.dataset}')
	for epoch in range(10):
		lossT, _ = backprop(model, next(iter(train_loader)), optimizer, scheduler)
	torch.zero_grad = True

	### Testing phase
	print(f'Testing {args.model} on {args.dataset}')
	data = next(iter(test_loader))
	loss, y_pred = backprop(model, data, optimizer, scheduler, training=False)

	### Plot curves
	plotter(f'{args.model}_{args.dataset}', data, y_pred, loss, labels)

	### Scores
	df = pd.DataFrame()
	lossT, _ = backprop(model, next(iter(train_loader)), optimizer, scheduler, training=False)
	for i in range(loss.shape[1]):
		lt, l, ls = lossT[:, i], loss[:, i], labels[:, i]
		result = pot_eval(lt, l, ls[1:])
		df = df.append(result, ignore_index=True)
	print(df)
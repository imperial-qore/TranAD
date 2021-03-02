import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
from src.constants import *
torch.manual_seed(1)

## Separate LSTM for each variable
class LSTM_Univariate(nn.Module):
	def __init__(self, feats):
		super(LSTM_Univariate, self).__init__()
		self.name = 'LSTM_Univariate'
		self.lr = 0.002
		self.n_feats = feats
		self.n_hidden = 1
		self.lstm = nn.ModuleList([nn.LSTM(1, self.n_hidden) for i in range(feats)])

	def forward(self, x):
		hidden = [(torch.rand(1, 1, self.n_hidden, dtype=torch.float64), 
			torch.randn(1, 1, self.n_hidden, dtype=torch.float64)) for i in range(self.n_feats)]
		outputs = []
		for i, g in enumerate(x):
			multivariate_output = []
			for j in range(self.n_feats):
				univariate_input = g.view(-1)[j].view(1, 1, -1)
				out, hidden[j] = self.lstm[j](univariate_input, hidden[j])
				multivariate_output.append(2 * out.view(-1))
			output = torch.cat(multivariate_output)
			outputs.append(output)
		return torch.stack(outputs)

## Single LSTM model for all variables
class LSTM_Multivariate(nn.Module):
	def __init__(self, feats):
		super(LSTM_Multivariate, self).__init__()
		self.name = 'LSTM_Multivariate'
		self.lr = 0.002
		self.n_feats = feats
		self.n_hidden = 64
		self.lstm = nn.LSTM(feats, self.n_hidden)
		self.lstm2 = nn.LSTM(feats, self.n_feats)
		# self.fcn = nn.Sequential(nn.Linear(self.n_hidden, self.n_feats), nn.Sigmoid())

	def forward(self, x):
		hidden = (torch.rand(1, 1, self.n_hidden, dtype=torch.float64), torch.randn(1, 1, self.n_hidden, dtype=torch.float64))
		hidden2 = (torch.rand(1, 1, self.n_feats, dtype=torch.float64), torch.randn(1, 1, self.n_feats, dtype=torch.float64))
		outputs = []
		for i, g in enumerate(x):
			out, hidden = self.lstm(g.view(1, 1, -1), hidden)
			out, hidden2 = self.lstm2(g.view(1, 1, -1), hidden2)
			# out = self.fcn(out.view(-1))
			outputs.append(2 * out.view(-1))
		return torch.stack(outputs)

## Single LSTM model for all variables + VAE
class LSTM_VAE(nn.Module):
	def __init__(self, feats):
		super(LSTM_VAE, self).__init__()
		self.name = 'LSTM_VAE'
		self.lr = 0.002
		self.beta = 0.01
		self.n_feats = feats
		self.n_hidden = 32
		self.n_latent = 8
		self.lstm = nn.GRU(feats, self.n_hidden, 2)
		self.encoder = nn.Sequential(
			nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
			nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
			nn.Flatten(),
			nn.Linear(self.n_hidden, 2*self.n_latent)
		)
		self.decoder = nn.Sequential(
			nn.Linear(self.n_latent, self.n_hidden), nn.PReLU(),
			nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
			nn.Linear(self.n_hidden, self.n_feats), nn.Sigmoid(),
		)

	def forward(self, x, hidden = None):
		hidden = torch.rand(2, 1, self.n_hidden, dtype=torch.float64) if hidden is not None else hidden
		out, hidden = self.lstm(x.view(1, 1, -1), hidden)
		## Encode
		x = self.encoder(out)
		mu, logvar = torch.split(x, [self.n_latent, self.n_latent], dim=-1)
		## Reparameterization trick
		std = torch.exp(0.5*logvar)
		eps = torch.randn_like(std)
		x = mu + eps*std
		## Decoder
		x = self.decoder(x)
		return x.view(-1), mu.view(-1), logvar.view(-1), hidden

## USAD Model (KDD 20)
class USAD(nn.Module):
	def __init__(self, feats):
		super(USAD, self).__init__()
		self.name = 'USAD'
		self.lr = 0.0001
		self.n_feats = feats
		self.n_hidden = 16
		self.n_latent = 5
		self.n_window = w_size
		self.n = self.n_feats * self.n_window
		self.encoder = nn.Sequential(
			nn.Flatten(),
			nn.Linear(self.n, self.n_hidden), nn.ReLU(),
			nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(),
			nn.Linear(self.n_hidden, self.n_latent), nn.ReLU(),
		)
		self.decoder1 = nn.Sequential(
			nn.Linear(self.n_latent,self.n_hidden), nn.ReLU(),
			nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(),
			nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
		)
		self.decoder2 = nn.Sequential(
			nn.Linear(self.n_latent,self.n_hidden), nn.ReLU(),
			nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(),
			nn.Linear(self.n_hidden, self.n), nn.Sigmoid(),
		)

	def forward(self, g):
		## Encode
		z = self.encoder(g.view(1,-1))
		## Decoders (Phase 1)
		ae1 = self.decoder1(z)
		ae2 = self.decoder2(z)
		## Encode-Decode (Phase 2)
		ae2ae1 = self.decoder2(self.encoder(ae1))
		return ae1.view(-1), ae2.view(-1), ae2ae1.view(-1)


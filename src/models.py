import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
torch.manual_seed(1)

class LSTM_Multivariate(nn.Module):
	def __init__(self, feats):
		super(LSTM_Multivariate, self).__init__()
		self.name = 'LSTM_Multivariate'
		self.lr = 0.005
		self.n_feats = feats
		self.n_hidden = 64
		self.lstm = nn.LSTM(feats, self.n_hidden)
		self.lstm2 = nn.LSTM(self.n_hidden, self.n_hidden)
		self.lstm3 = nn.LSTM(self.n_hidden, feats)

	def forward(self, x):
		hidden = (torch.rand(1, 1, self.n_hidden, dtype=torch.float64), torch.randn(1, 1, self.n_hidden, dtype=torch.float64))
		hidden2 = (torch.rand(1, 1, self.n_hidden, dtype=torch.float64), torch.randn(1, 1, self.n_hidden, dtype=torch.float64))
		hidden3 = (torch.rand(1, 1, self.n_feats, dtype=torch.float64), torch.randn(1, 1, self.n_feats, dtype=torch.float64))
		outputs = []
		for i, g in enumerate(x):
			out, hidden = self.lstm(g.view(1, 1, -1), hidden)
			out, hidden2 = self.lstm2(out.view(1, 1, -1), hidden2)
			out, hidden3 = self.lstm3(out.view(1, 1, -1), hidden3)
			outputs.append(2 * out.view(-1))
		return torch.stack(outputs)


class LSTM_VAE(nn.Module):
	def __init__(self, feats):
		super(LSTM_VAE, self).__init__()
		self.name = 'LSTM_VAE'
		self.lr = 0.005
		self.beta = 0.01
		self.n_feats = feats
		self.n_hidden = 64
		self.n_latent = 16
		self.lstm = nn.LSTM(feats, self.n_hidden)
		self.encoder = nn.Sequential(
			nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
			nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
			nn.Flatten(),
			nn.Linear(self.n_hidden, 2*self.n_latent)
		)
		self.decoder = nn.Sequential(
			nn.Linear(self.n_latent, self.n_hidden), nn.PReLU(),
			nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
			nn.Linear(self.n_hidden, self.n_feats), nn.Tanh(),
		)

	def forward(self, x, training):
		hidden = (torch.rand(1, 1, self.n_hidden, dtype=torch.float64), torch.randn(1, 1, self.n_hidden, dtype=torch.float64))
		outputs, mus, logvars = [], [], []
		for i, g in enumerate(x):
			out, hidden = self.lstm(g.view(1, 1, -1), hidden)
			## Encode
			x = self.encoder(out)
			mu, logvar = torch.split(x, [self.n_latent, self.n_latent], dim=-1)
			## Reparameterization trick
			std = torch.exp(0.5*logvar)
			eps = torch.randn_like(std)
			x = mu + eps*std if training else mu
			## Decoder
			x = 4 * self.decoder(x)
			outputs.append(x.view(-1))
			mus.append(mu.view(-1))
			logvars.append(logvar.view(-1))
		return torch.stack(outputs), torch.stack(mus), torch.stack(logvars)


import torch
import torch.nn as nn
import torch.nn.functional as F

class R_NVP(nn.Module):
    def __init__(self, d, k, hidden, base_dist, even_odd=False):
        super().__init__()
        self.d, self.k = d, k
        self.base_dist = base_dist
        self.even_odd = even_odd
        self.sig_net = nn.Sequential(
                  nn.Linear(k, hidden),
                  nn.LeakyReLU(),
                  nn.Linear(hidden, d - k))
        self.mu_net = nn.Sequential(
                  nn.Linear(k, hidden),
                  nn.LeakyReLU(),
                  nn.Linear(hidden, d - k))
        
    def forward(self, x, flip=False):
        #print(x.shape)
        if not self.even_odd:
          x1, x2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:]
        else:
          x1, x2 = x[:, ::2], x[:, 1::2]

        if flip:
            x2, x1 = x1, x2

        sig = self.sig_net(x1)
        mu = self.mu_net(x1)
        z1, z2 = x1, x2 * torch.exp(sig) + mu

        if flip:
            z2, z1 = z1, z2
        z_hat = torch.cat([z1, z2], dim=-1)
        log_jacob = sig.sum(-1)

        return z_hat, log_jacob

    def inverse(self, Z, flip=False):
        z1, z2 = Z[:, :Z.shape[1]//2], Z[:, Z.shape[1]//2:]

        if flip:
            z2, z1 = z1, z2

        x1 = z1
        x2 = (z2 - self.mu_net(z1)) * torch.exp(-self.sig_net(z1))

        if flip:
            x2, x1 = x1, x2
        if not self.even_odd:
          return torch.cat([x1, x2], -1)
        else:
          temp1, temp2 = torch.zeros_like(torch.cat([x1, x2], -1)), torch.zeros_like(torch.cat([x1, x2], -1))
          temp1[:,::2] = x1
          temp2[:,1::2] = x2
          return temp1 + temp2

class stacked_NVP(nn.Module):
    def __init__(self, d, k, hidden, n, base_dist, even_odd=False):
        super().__init__()
        self.bijectors = nn.ModuleList([
            R_NVP(d, k+((i%2)*(d%2)), hidden=hidden, base_dist=base_dist, even_odd=even_odd) for i in range(n)
        ])
        self.flips = [True if i%2 else False for i in range(n)]

    def forward(self, x):
        log_jacobs = []

        for bijector, f in zip(self.bijectors, self.flips):
            x, lj = bijector(x, flip=self.flips[f])
            log_jacobs.append(lj)

        return x, sum(log_jacobs)

    def inverse(self, z):
        for bijector, f in zip(reversed(self.bijectors), reversed(self.flips)):
            z = bijector.inverse(z, flip=self.flips[f])
        return z

import torch
import torch.nn as nn
import torch.nn.functional as F

class NICE(nn.Module) :

  def __init__(self, d, k, hidden, base_dist):
    super().__init__()
    self.d, self.k = d, k
    self.net = nn.Sequential(
        nn.Linear(k, hidden),
        nn.LeakyReLU(),
        nn.Linear(hidden, d - k))
    self.base_dist = base_dist

  def forward(self, x, flip=False) :

    x1, x2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:]
    if flip:
      x2, x1 = x1, x2
    z1, z2 = x1, x2 + self.net(x1)

    if flip:
      z2, z1 = z1, z2
    z_hat = torch.cat([z1, z2], dim=-1)

    return z_hat, 1

  def inverse(self, z, flip=False):
    z1, z2 = z[:, :z.shape[1]//2], z[:, z.shape[1]//2:]

    if flip:
      z2, z1 = z1, z2

    x1 = z1
    x2 = z2 - self.net(z1)

    if flip:
      x2, x1 = x1, x2
    return torch.cat([x1, x2], -1)

class stacked_NICE(nn.Module):
    def __init__(self, d, k, hidden, n, base_dist):
        super().__init__()
        self.bijectors = nn.ModuleList([
            NICE(d, k+((i%2)*(d%2)), hidden=hidden, base_dist=base_dist) for i in range(n)
        ])
        self.flips = [True if i%2 else False for i in range(n)]
        self.base_dist = base_dist

    def forward(self, x):

        for bijector, f in zip(self.bijectors, self.flips):
          x, log_pz = bijector(x, flip=f)

        return x, torch.zeros_like(x[:,0])

    def inverse(self, z):
        for bijector, f in zip(reversed(self.bijectors), reversed(self.flips)):
          z = bijector.inverse(z, flip=f)

        return z

import torch
import torch.nn as nn
import torch.nn.functional as F

class NICE(nn.Module) :

  def __init__(self, d, k, hidden, base_dist, even_odd=False):
    super().__init__()
    self.d, self.k = d, k
    self.net = nn.Sequential(
        nn.Linear(k, hidden),
        nn.LeakyReLU(),
        nn.Linear(hidden, d - k))
    self.base_dist = base_dist
    self.even_odd = even_odd

  def forward(self, x, flip=False) :
    if not self.even_odd:
      x1, x2 = x[:, :x.shape[1]//2], x[:, x.shape[1]//2:]
    else:
      x1, x2 = x[:, ::2], x[:, 1::2]
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
    if not self.even_odd:
      return torch.cat([x1, x2], -1)
    else:
      temp1, temp2 = torch.zeros_like(torch.cat([x1, x2], -1)), torch.zeros_like(torch.cat([x1, x2], -1))
      temp1[:,::2] = x1
      temp2[:,1::2] = x2
      return temp1 + temp2

class stacked_NICE(nn.Module):
    def __init__(self, d, k, hidden, n, base_dist, even_odd=False):
        super().__init__()
        self.bijectors = nn.ModuleList([
            NICE(d, k+((i%2)*(d%2)), hidden=hidden, base_dist=base_dist, even_odd=even_odd) for i in range(n)
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

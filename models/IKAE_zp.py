import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from Sentinel2TS.models.NICE import NICE, stacked_NICE
from Sentinel2TS.models.RealNVP import R_NVP, stacked_NVP

class IKAE_zp(nn.Module):
    def __init__(self, input_dim:int, hidden_dim=64, n_layers_encoder=3, zero_padding=0,
                 even_odd=False, random_K=False, positive_nonlin=torch.abs, bounded=False, flow='RNVP', device='cpu'):
        super(IKAE_zp, self).__init__()


        # Encoder
        self.input_dim = input_dim
        self.zero_padding = zero_padding
        self.latent_dim = input_dim + zero_padding
        self.positive_nonlin = positive_nonlin
        self.bounded = bounded
        base_mu, base_cov = torch.zeros(self.latent_dim), torch.eye(self.latent_dim)
        base_dist = MultivariateNormal(base_mu, base_cov)
        if flow == 'RNVP':
          self.invertible_encoder = stacked_NVP(self.latent_dim, self.latent_dim // 2, hidden=hidden_dim, n=n_layers_encoder,
                                                base_dist=base_dist, even_odd=even_odd).to(device)
        elif flow == 'NICE':
          self.invertible_encoder = stacked_NICE(self.latent_dim, self.latent_dim // 2, hidden=hidden_dim, n=n_layers_encoder, base_dist=base_dist)
        else:
          raise Exception("""Argument "flow" should be "NICE" or "RNVP".""")

        # Koopman operator
        if not random_K:
          self.K = torch.eye(self.latent_dim, requires_grad=True, device=device)
        else:
          M = torch.randn((self.latent_dim, self.latent_dim))
          A = M - M.T
          self.K = torch.matrix_exp(A).to(device).clone().detach().requires_grad_()
          #print(self.K.is_leaf)
        self.state_dict()['K'] = self.K
        self.device = device

    def encode(self, x):
        """Encode input data x using the encoder layers."""
        #print(x.shape)
        x = torch.cat((x, torch.zeros((x.shape[0], self.zero_padding)).to(self.device)), dim=1)
        #print(x.shape)
        y, _ = self.invertible_encoder(x)
        return y

    def decode(self, x):
        """Decode the encoded data x using the decoder layers."""
        return self.invertible_encoder.inverse(x)[:,:self.input_dim]

    def one_step_ahead(self, x):
        """Predict one-step-ahead in the latent space using the Koopman operator."""
        return torch.matmul(self.K, x.T).T

    def one_step_back(self, x):
        """Predict one-step-back in the latent space using the inverse of the Koopman operator."""
        return torch.matmul(torch.inverse(self.K), x.T).T

    def forward(self, x):
        """
        Perform forward pass through the model.

        Args:
            x (torch.Tensor): Input state.

        Returns:
            x_advanced (torch.Tensor): Estimated state after one time step.
            phi (torch.Tensor): Encoded input state.
            phi_advanced (torch.Tensor): Encoded input state advanced by one time step.
        """
        phi = self.encode(x)
        phi_advanced = self.one_step_ahead(phi)
        x_advanced = self.decode(phi_advanced)
        return x_advanced, phi, phi_advanced

    def forward_n(self, x, n):
        """
        Perform forward pass for n steps.

        Args:
            x (torch.Tensor): Input state.
            n (int): Number of steps to advance.

        Returns:
            x_advanced (torch.Tensor): Estimated state after n time steps.
            phi (torch.Tensor): Encoded input state.
            phi_advanced (torch.Tensor): Encoded state advanced by n time steps.
        """
        phi = self.encode(x)
        phi_advanced = self.one_step_ahead(phi)
        for k in range(n-1):
            phi_advanced = self.one_step_ahead(phi_advanced)
        x_advanced = self.decode(phi_advanced)
        return x_advanced, phi, phi_advanced

    def forward_n_remember(self, x, n, training=False, center=False):
        """
        Perform forward pass for n steps while remembering intermediate latent states.

        Args:
            x (torch.Tensor): Input state.
            n (int): Number of steps to advance.
            training (bool, optional): Flag to indicate training mode (default: False).

        Returns:
            x_advanced (torch.Tensor or None): Estimated state after n time steps if not training, otherwise None.
            phis (torch.Tensor): Encoded state at each step, concatenated along the 0th dimension.
        """
        phi = self.encode(x)
        phis = [phi]
        for k in range(n):
            phis.append(self.one_step_ahead(phis[-1]))
        x_advanced = None if training else self.decode(phis[n])
        return x_advanced, torch.cat(tuple(phi.unsqueeze(0) for phi in phis), dim=0)

    def forward_n_multiple(self, x, n, n_samples, scale=1):
      mu_invertible, mu_augmentation, var = self.encode(x)
      phis = [self.sample_state_multiple(mu_invertible, mu_augmentation, var, n_samples, scale=scale, print_shape=False)]
      for k in range(n):
        if len(phis[-1].shape) == 2:
          phis.append(self.one_step_ahead(phis[-1]))
        else:
          phis_shape = phis[-1].shape
          phis.append(self.one_step_ahead(phis[-1].flatten(0,1)))
          phis[-1] = phis[-1].reshape((phis_shape[0], phis_shape[1], phis_shape[2]))
      if len(phis[n].shape) == 2:
        x_advanced = self.decode(phis[n])
      else:
        x_advanced = self.decode(phis[n].flatten(0,1)).reshape(phis[n].shape[0], phis[n].shape[1], model.input_dim)
      return x_advanced, torch.cat(tuple(phi.unsqueeze(0) for phi in phis), dim=0).transpose(0,1)


    def backward(self, x):
        """
        Perform backward pass through the model.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            x_advanced (torch.Tensor): Estimated state after one step back.
            phi (torch.Tensor): Encoded input state.
            phi_advanced (torch.Tensor): Encoded state advanced one step back.
        """
        phi = self.encode(x)
        phi_advanced = self.one_step_back(phi)
        x_advanced = self.decode(phi_advanced)
        return x_advanced, phi, phi_advanced

    def backward_n(self, x, n):
        """
        Perform backward pass for n steps.

        Args:
            x (torch.Tensor): Input state.
            n (int): Number of steps to advance.

        Returns:
            x_advanced (torch.Tensor): Estimated state after n steps back.
            phi (torch.Tensor): Encoded input state.
            phi_advanced (torch.Tensor): Encoded state advanced n steps back.
        """
        phi = self.encode(x)
        phi_advanced = self.one_step_back(phi)
        for k in range(n-1):
            phi_advanced = self.one_step_ahead(phi_advanced)
        x_advanced = self.decode(phi_advanced)
        return x_advanced, phi, phi_advanced

    def backward_n_remember(self, x, n):
        """
        Perform backward pass for n steps while remembering intermediate states.

        Args:
            x (torch.Tensor): Input state.
            n (int): Number of steps to advance.

        Returns:
            x_advanced (torch.Tensor): Reconstructed state after n steps back.
            phis (torch.Tensor): Encoded state at each step, concatenated along the 0th dimension.
        """
        phis = []
        phis.append(self.encode(x))
        for k in range(n):
            phis.append(self.one_step_back(phis[-1]))
        x_advanced = self.decode(phis[n])
        return x_advanced, torch.cat(tuple(phi.unsqueeze(0) for phi in phis), dim=0)

    def configure_optimizers(self, lr=1e-3, K_lr=None, weight_decay=0):
        """
        Configure the optimizer for training the model.

        Args:
            lr (float, optional): Learning rate for the optimizer (default: 1e-3).

        Returns:
            torch.optim.Optimizer: Optimizer instance.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        if K_lr is None:
          K_lr = lr
        optimizer.add_param_group({"params": self.K, "lr": K_lr})
        return optimizer

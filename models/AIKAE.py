import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
from Sentinel2TS.models.NICE import NICE, stacked_NICE
from Sentinel2TS.models.RealNVP import R_NVP, stacked_NVP

class AIKAE(nn.Module):
    def __init__(self, input_dim:int, hidden_dim=64, n_layers_encoder=3, augmentation_dims=[256,128, 16],
                 even_odd=False, random_K=False, positive_nonlin=torch.abs, flow='RNVP', device='cpu'):
        """
        Koopman Autoencoder class, comprising an auto-encoder and a Koopman matrix.

        Args:
            input_dim (int): Dimension of the input data.
            linear_dims (list): List of linear layer dimensions.
            device (str, optional): Device to run the model on (default: 'cpu').
        """
        super(AIKAE, self).__init__()


        # Encoder
        self.input_dim = input_dim
        self.augmentation_dim = augmentation_dims[-1]
        self.latent_dim = input_dim + augmentation_dims[-1]
        self.positive_nonlin = positive_nonlin
        base_mu, base_cov = torch.zeros(self.latent_dim), torch.eye(self.latent_dim)
        base_dist = MultivariateNormal(base_mu, base_cov)
        if flow == 'RNVP':
          self.invertible_encoder = stacked_NVP(input_dim, input_dim // 2, hidden=hidden_dim, n=n_layers_encoder,
                                                base_dist=base_dist, even_odd=even_odd).to(device)
        elif flow == 'NICE':
          self.invertible_encoder = stacked_NICE(input_dim, input_dim // 2, hidden=hidden_dim, n=n_layers_encoder, base_dist=base_dist)
        else:
          raise Exception("""Argument "flow" should be "NICE" or "RNVP".""")
        self.augmentation_encoder = nn.ModuleList()
        self.augmentation_encoder.add_module("encoder_1", nn.Linear(input_dim, augmentation_dims[0]))
        for i in range(len(augmentation_dims)-2):
            self.augmentation_encoder.add_module(f"encoder_{i+2}", nn.Linear(augmentation_dims[i], augmentation_dims[i+1]))
        self.augmentation_encoder.add_module(f"encoder_{len(augmentation_dims)}", nn.Linear(augmentation_dims[len(augmentation_dims)-2], self.augmentation_dim))

        # Koopman operator
        if not random_K:
          self.K = torch.eye(self.latent_dim, requires_grad=True, device=device)
        else:
          M = torch.randn((self.latent_dim, self.latent_dim))
          A = M - M.T
          self.K = torch.matrix_exp(A).to(device).clone().detach().requires_grad_()
          print(self.K.is_leaf)
        self.state_dict()['K'] = self.K

    def encode(self, x):
        """Encode input data x using the encoder layers."""
        invertible_part, _ = self.invertible_encoder(x)
        augmentation_part = x
        for layer_idx, layer in enumerate(self.augmentation_encoder):
            augmentation_part = layer(augmentation_part)
            if layer_idx < len(self.augmentation_encoder) - 1:
                augmentation_part = F.relu(augmentation_part)
        return torch.cat([invertible_part, augmentation_part], dim=-1)

    def log_jacob(self, x):
        _, output = self.invertible_encoder(x)
        return output

    def decode(self, x):
        """Decode the encoded data x using the decoder layers."""
        return self.invertible_encoder.inverse(x[:, :self.input_dim])

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

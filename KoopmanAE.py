import torch
import torch.nn as nn
import torch.nn.functional as F

class KoopmanAE(nn.Module):
    def __init__(self, input_dim:int, linear_dims:list, device='cpu'):
        """
        Koopman Autoencoder class, comprising an auto-encoder and a Koopman matrix.

        Args:
            input_dim (int): Dimension of the input data.
            linear_dims (list): List of linear layer dimensions.
            device (str, optional): Device to run the model on (default: 'cpu').
        """
        super(KoopmanAE, self).__init__()

        self.latent_dim = linear_dims[-1]

        # Encoder layers
        self.encoder = nn.ModuleList()
        self.encoder.add_module("encoder_1", nn.Linear(input_dim, linear_dims[0]))
        for i in range(len(linear_dims)-1):
            self.encoder.add_module(f"encoder_{i+2}", nn.Linear(linear_dims[i], linear_dims[i+1]))

        # Decoder layers
        self.decoder = nn.ModuleList()
        for i in range(len(linear_dims)-1):
            self.decoder.add_module(f"decoder_{i+1}", nn.Linear(linear_dims[-i-1], linear_dims[-i-2]))
        self.decoder.add_module(f"decoder_{len(linear_dims)}", nn.Linear(linear_dims[0], input_dim))

        # Koopman operator
        self.K = torch.eye(self.latent_dim, requires_grad=True, device=device)
        self.state_dict()['K'] = self.K

    def encode(self, x):
        """Encode input data x using the encoder layers."""
        for layer_idx, layer in enumerate(self.encoder):
            x = layer(x)
            if layer_idx < len(self.encoder) - 1:
                x = F.relu(x)
        return x

    def decode(self, x):
        """Decode the encoded data x using the decoder layers."""
        for layer_idx, layer in enumerate(self.decoder):
            x = layer(x)
            if layer_idx < len(self.decoder) - 1:
                x = F.relu(x)
        return x

    def one_step_ahead(self, x):
        """Predict one-step-ahead in the latent space using the Koopman operator."""
        return torch.matmul(x, self.K)

    def one_step_back(self, x):
        """Predict one-step-back in the latent space using the inverse of the Koopman operator."""
        return torch.matmul(x, torch.inverse(self.K))

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

    def forward_n_remember(self, x, n, training=False):
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
        phis = []
        phis.append(self.encode(x))
        for k in range(n):
            phis.append(self.one_step_ahead(phis[-1]))
        x_advanced = None if training else self.decode(phis[n])
        return x_advanced, torch.cat(tuple(phi.unsqueeze(0) for phi in phis), dim=0)

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
            phi_advanced = self.one_step_back(phi_advanced)
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
        optimizer.add_param_group({"params": self.K})
        return optimizer

    def mse_loss(self, x, x_hat):
        """
        Calculate the Mean Squared Error (MSE) loss between input and reconstructed data.

        Args:
            x (torch.Tensor): Input data.
            x_hat (torch.Tensor): Reconstructed data.

        Returns:
            torch.Tensor: Mean squared error loss.
        """
        total_prediction_loss = torch.sum((x-x_hat)**2)/(x.size()[0])
        return total_prediction_loss

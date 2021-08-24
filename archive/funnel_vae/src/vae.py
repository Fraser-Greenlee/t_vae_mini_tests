from transformers.utils import logging
import torch
from torch import nn

from funnel_vae.src.outputs import BaseVaeOutput

logger = logging.get_logger(__name__)


class VAE(nn.Module):
    """
    An MMD-VAE used with encoder-decoder models.
    Encodes all token encodings into a single latent & spits them back out.
    """

    batch_size = None

    def __init__(self, encoder, decoder, use_reg_loss=True):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.use_reg_loss = use_reg_loss

    def _model_forward(self, encoding, latent=None, attention_mask=None):
        if latent is None:
            latent = self.encoder(encoding, attention_mask=attention_mask)
        return self.decoder(latent), latent

    def forward(
        self,
        input_encoding=None,
        latent=None,
        skip_reg_loss=False,
        attention_mask=None,
    ):
        if input_encoding is None and latent is None:
            raise ValueError("Both `input_encoding` and `latent` sent to VAE are None.")
        use_reg_loss = self.use_reg_loss and latent is None and skip_reg_loss is False  # don't regularise if given latent
        recon_encoding, latent = self._model_forward(input_encoding, latent=latent, attention_mask=attention_mask)
        if use_reg_loss:
            # treat each token encoding as a seperate latent code
            batch_size, n_latents_per_batch, latent_code_dim = latent.size()
            reg_loss = self._regularliser_loss(latent.reshape(-1, latent_code_dim)) / batch_size * n_latents_per_batch
        else:
            reg_loss = torch.tensor(0, device=latent.device)
        return BaseVaeOutput(latent=latent, reconstructed_encoding=recon_encoding, reg_loss=reg_loss)

    @staticmethod
    def _compute_kernel(x, y):
        x_size = x.shape[0]
        y_size = y.shape[0]
        dim = x.shape[1]

        tiled_x = x.view(x_size, 1, dim).repeat(1, y_size, 1)
        tiled_y = y.view(1, y_size, dim).repeat(x_size, 1, 1)

        return torch.exp(-torch.mean((tiled_x - tiled_y) ** 2, dim=2) / dim * 1.0)

    def _compute_mmd(self, x, y):
        x_kernel = self._compute_kernel(x, x)
        y_kernel = self._compute_kernel(y, y)
        xy_kernel = self._compute_kernel(x, y)
        return torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)

    def _regularliser_loss(self, latent):
        true_samples = torch.randn(latent.size(), device=latent.device)
        return self._compute_mmd(true_samples, latent)

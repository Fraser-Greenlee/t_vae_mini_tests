from transformers import EncoderDecoderConfig


class TransformerVaeConfig(EncoderDecoderConfig):
    model_type = "vae"
    is_composition = True

    def __init__(
        self,

        d_input=None,
        d_output=None,
        latent_size=32,

        vae_encoder_n_layers=1,
        vae_encoder_use_dropout=False,

        vae_decoder_n_layers=1,
        vae_decoder_use_dropout=False,

        **kwargs
    ):
        super().__init__(**kwargs)

        self.d_input = d_input
        self.d_output = d_output
        self.latent_size = latent_size

        self.vae_encoder_n_layers = vae_encoder_n_layers
        self.vae_encoder_use_dropout = vae_encoder_use_dropout

        self.vae_decoder_n_layers = vae_decoder_n_layers
        self.vae_decoder_use_dropout = vae_decoder_use_dropout

        self.is_encoder_decoder = True

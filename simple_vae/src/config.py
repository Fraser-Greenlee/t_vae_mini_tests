from transformers import EncoderDecoderConfig


class TransformerVaeConfig(EncoderDecoderConfig):
    model_type = "vae"
    is_composition = True

    def __init__(
        self,

        d_input=None,
        d_output=None,
        latent_size=32,

        encoder_n_layers=1,
        encoder_use_dropout=False,

        decoder_n_layers=1,
        decoder_use_dropout=False,

        **kwargs
    ):
        super().__init__(**kwargs)

        self.d_input = d_input
        self.d_output = d_output
        self.latent_size = latent_size

        self.encoder_n_layers = encoder_n_layers
        self.encoder_use_dropout = encoder_use_dropout

        self.decoder_n_layers = decoder_n_layers
        self.decoder_use_dropout = decoder_use_dropout

        self.is_encoder_decoder = True

from transformers import trainer as trainer_script
from transformers.utils import logging
from transformers.integrations import (
    WandbCallback,
    is_wandb_available,
    TensorBoardCallback,
    CometCallback,
    AzureMLCallback,
    MLflowCallback,
)

from funnel_vae.src.trainer_callback import WandbCallbackUseModelLogs, TellModelGlobalStep

logger = logging.get_logger(__name__)

assert(is_wandb_available())

NOT_ALLOWED_LOGGERS = [TensorBoardCallback, CometCallback, AzureMLCallback, MLflowCallback]

for logger_integration in NOT_ALLOWED_LOGGERS:
    removed = []
    if logger_integration in trainer_script.DEFAULT_CALLBACKS:
        trainer_script.DEFAULT_CALLBACKS.remove(logger_integration)
        removed.append(logger_integration)
    logger.info(f"Only supports W&B logging, removed loggers: {removed}")


class VaeTrainer(trainer_script.Trainer):
    text_to_array = None

    def __init__(self, model=None, args=None, custom_methods={}, **kwargs):
        super().__init__(model, args, **kwargs)
        self.remove_callback(WandbCallback)
        self.add_callback(WandbCallbackUseModelLogs)
        self.add_callback(TellModelGlobalStep)


import logging
from dataclasses import dataclass, field

from transformers import (
    TrainingArguments,
)

logger = logging.getLogger(__name__)


@dataclass
class VaeTrainingArguments(TrainingArguments):
    pass

# Python Standard Library
from pathlib import Path

# Third Party Libraries
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger


class Trainer(L.Trainer):
  def __init__(self, seed):
    L.seed_everything(seed)
    logger = TensorBoardLogger(Path('/workspace/code/data/logs'))
    early_stopping = EarlyStopping(monitor="val/acc_epoch", mode="max", verbose=True)
    super().__init__(
        max_epochs=30,
        devices="auto",
        accelerator="auto",
        logger=logger,
        default_root_dir=Path('/workspace/code/data/logs'),
        callbacks=[early_stopping],
    )

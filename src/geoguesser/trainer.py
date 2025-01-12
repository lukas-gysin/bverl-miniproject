# Python Standard Library
from pathlib import Path

# Third Party Libraries
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger


class Trainer(L.Trainer):
  def __init__(self, seed, experiment_name):
    L.seed_everything(seed)
    logger = CSVLogger(Path('/workspace/code/data/logs'), name=experiment_name, version=0)
    early_stopping = EarlyStopping(monitor="val/acc_epoch", mode="max", patience=5)
    super().__init__(
        max_epochs=50,
        devices="auto",
        accelerator="auto",
        logger=logger,
        default_root_dir=Path('/workspace/code/data/logs'),
        callbacks=[early_stopping],
    )

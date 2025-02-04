# Third Party Libraries
import lightning as L
import torch
import torch.nn as nn
import torchmetrics


class Classifier(L.LightningModule):
    def __init__(self, model, n_classes: int = 10, weight_decay: float = 0.01, learning_rate:float = 0.001):
        super().__init__()
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=n_classes)
        self.train_loss = torchmetrics.MeanMetric()
        self.val_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=n_classes)
        self.val_loss = torchmetrics.MeanMetric()
        self.test_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=n_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        logits = self(x)

        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)

        # Update accuracy metric
        self.train_accuracy(preds, y)
        self.train_loss(loss)


        self.log("train/acc_step", self.train_accuracy, prog_bar=True)
        self.log("train/loss_step", loss, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        # log epoch metric
        self.log("train/acc_epoch", self.train_accuracy)
        self.log("train/loss_epoch", self.train_loss)

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)

        # Update accuracy metric
        self.val_accuracy(preds, y)
        self.val_loss(loss)

        self.log("val/acc_step", self.val_accuracy, prog_bar=True)
        self.log("val/loss_step", self.val_loss, prog_bar=True)

        return loss

    def on_validation_epoch_end(self):
      self.log("val/acc_epoch", self.val_accuracy)
      self.log("val/loss_epoch", self.val_loss)

    def test_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)

        # Update accuracy metric
        self.test_accuracy(preds, y)

        self.log("test/acc_step", self.test_accuracy)

        return loss

    def on_test_epoch_end(self):
        # log epoch metric
        self.log("test/acc_epoch", self.test_accuracy)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

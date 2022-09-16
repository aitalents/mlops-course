import torch
from config import AppConfig
from pytorch_lightning import Trainer, seed_everything, LightningModule
from torchvision import models, datasets
from PIL import Image
import numpy as np
import cv2
class LigthingModel(LightningModule):
    def __init__(self, model, optimizer, loss_func, lr):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.lr = lr

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        logits = self.model(x)
        loss = self.loss_func(logits, y)
        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        return optimizer

def main_actions(config: AppConfig):
    seed_everything(42, workers=True)
    trainer = Trainer(max_epochs=5, accelerator="gpu")

    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = torch.nn.Linear(num_ftrs, 2)
    pl_model = LigthingModel(model_ft, torch.optim.Adam, torch.nn.CrossEntropyLoss(), 0.001)
    image_datasets = {x: datasets.ImageFolder(config.training_dataset_path/x, loader=lambda x: torch.Tensor(cv2.imread(x)).view(3,512,512))
                    for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                shuffle=True, num_workers=4)
                for x in ['train', 'val']}


    trainer.fit(pl_model, dataloaders["train"], dataloaders["val"])


def main():
    pass


if __name__ == "__main__":
    main()

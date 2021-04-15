import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

import torchvision.models as models

from azureml.core import Run
run = Run.get_context()

class ResNetClassificationModel(pl.LightningModule):
    def __init__(self):
        super(ResNetClassificationModel, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18.fc = nn.Linear(512, 10)
        
        self.val_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

    def forward(self, x):
        y = self.resnet18(x)
        y = F.softmax(y,dim=1)
        return y

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        
        run.log("loss", float(loss))
        #self.log("loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_nb): 
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        preds = torch.argmax(y, dim=1)
        
        run.log("val_loss", float(loss))
        run.log("val_acc", float(self.val_acc(y,t)))
        #self.log('val_loss', loss, prog_bar=True)
        #self.log('val_acc', self.val_acc(y,t), prog_bar=True)
        return loss

    def test_step(self, batch, batch_nb):
        x, t = batch
        y = self(x)
        loss = F.cross_entropy(y, t)
        preds = torch.argmax(y, dim=1)
        
        run.log("test_loss", float(loss))
        run.log("test_acc", float(self.test_acc(y,t)))
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

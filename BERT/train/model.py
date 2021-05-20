import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from transformers import BertJapaneseTokenizer
from transformers import BertModel

from azureml.core import Run
run = Run.get_context()

class BERTClassificationModel(pl.LightningModule):
    def __init__(self):
        super(BERTClassificationModel, self).__init__()
        self.bert = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
        self.output = nn.Linear(768, 9)
        
        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

    def forward(self, input_ids, attention_mask, token_type_ids):
        y = self.bert(input_ids, attention_mask, token_type_ids).last_hidden_state
        ## cls token相当部分のhidden_stateのみ抜粋
        y = y[:,0,:]
        y = y.view(-1, 768)
        y = self.output(y)
        y = F.softmax(y, dim=1)
        return y

    def training_step(self, batch, batch_nb):
        x, t = batch
        y = self(x['input_ids'], x['attention_mask'], x['token_type_ids'])
        loss = F.cross_entropy(y, t)
        run.log("loss", float(loss))
        ##self.log("loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_nb): 
        x, t = batch
        y = self(x['input_ids'], x['attention_mask'], x['token_type_ids'])
        loss = F.cross_entropy(y, t)
        preds = torch.argmax(y, dim=1)
        run.log("val_loss", float(loss))
        run.log("val_acc", float(self.val_acc(y,t)))
        ##self.log('val_loss', loss, prog_bar=True)
        ##self.log('val_acc', self.val_acc(y,t), prog_bar=True)
        return loss

    def test_step(self, batch, batch_nb):
        x, t = batch
        y = self(x['input_ids'], x['attention_mask'], x['token_type_ids'])
        loss = F.cross_entropy(y, t)
        preds = torch.argmax(y, dim=1)
        run.log("test_loss", float(loss))
        run.log("test_acc", float(self.test_acc(y,t)))
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam([
            {'params': self.bert.encoder.layer[-1].parameters(), 'lr': 5e-5},
            {'params': self.output.parameters(), 'lr': 1e-4}
        ])

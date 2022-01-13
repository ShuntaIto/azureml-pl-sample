import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics

from transformers import BertJapaneseTokenizer
from transformers import BertModel

from azureml.core import Run
run = Run.get_context()

class BERTClassificationModel(pl.LightningModule):
    def __init__(self, bert_lr=5e-5, output_lr=1e-5):
        super(BERTClassificationModel, self).__init__()
        self.bert = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
        self.output = nn.Linear(768, 9)
        
        self.bert_lr = bert_lr
        self.output_lr = output_lr
        
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def forward(self, input_ids, attention_mask, token_type_ids):
        y = self.bert(input_ids, attention_mask, token_type_ids).last_hidden_state
        y = y.type_as(y)
        ## cls token相当部分のhidden_stateのみ抜粋
        y = y[:,0,:].type_as(y)
        y = y.view(-1, 768).type_as(y)
        y = self.output(y).type_as(y)
        y = F.softmax(y, dim=1)
        return y

    def training_step(self, batch, batch_nb):
        x, t = batch
        y = self(x['input_ids'], x['attention_mask'], x['token_type_ids'])
        loss = F.cross_entropy(y, t)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

    def training_epoch_end(self, training_step_outputs):
        avg_loss = torch.stack([x['loss'] for x in self.all_gather(training_step_outputs)]).mean()
        print(self.all_gather(training_step_outputs))
        if self.trainer.is_global_zero:
            run.log("train_loss", float(avg_loss))
    
    def validation_step(self, batch, batch_nb): 
        x, t = batch
        y = self(x['input_ids'], x['attention_mask'], x['token_type_ids'])
        loss = F.cross_entropy(y, t)
        #preds = torch.argmax(y, dim=1)
        return {'val_loss': loss, 'val_acc': self.val_acc(y,t)}

    def validation_epoch_end(self, validation_step_outputs):
        avg_loss = torch.stack([x['val_loss'] for x in self.all_gather(validation_step_outputs)]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in self.all_gather(validation_step_outputs)]).mean()
        if self.trainer.is_global_zero:
            run.log("val_loss", float(avg_loss))
            run.log("val_acc", float(avg_acc))
    
    def test_step(self, batch, batch_nb):
        x, t = batch
        y = self(x['input_ids'], x['attention_mask'], x['token_type_ids'])
        loss = F.cross_entropy(y, t)
        #preds = torch.argmax(y, dim=1)
        return {'test_acc': self.test_acc(y,t)}

    def test_epoch_end(self, test_step_outputs):
        avg_acc = torch.stack([x['test_acc'] for x in self.all_gather(test_step_outputs)]).mean()
        if self.trainer.is_global_zero:
            run.log("test_acc", float(avg_acc))
    
    def configure_optimizers(self):
        return torch.optim.Adam([
            {'params': self.bert.encoder.layer[-1].parameters(),
             'lr': self.bert_lr},
            {'params': self.output.parameters(), 'lr': self.output_lr}
        ])

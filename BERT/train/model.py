import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics

from transformers import BertModel
from transformers import AdamW

class BERTClassificationModel(pl.LightningModule):
    def __init__(self, num_classes=9, bert_lr=5e-5, output_lr=1e-4, fix_bert=True):
        super(BERTClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.bert_lr = bert_lr
        self.output_lr = output_lr

        self.bert = BertModel.from_pretrained(
            'cl-tohoku/bert-base-japanese-whole-word-masking',
            return_dict=True,
        )
        self.output = nn.Linear(self.bert.config.hidden_size, 9)
        self.softmax = nn.Softmax(dim=1)

        if fix_bert:
            # fix BERT model 
            for param in self.bert.parameters():
                param.requires_grad = False
            # unfix BERT final layer
            for param in self.bert.encoder.layer[-1].parameters():
                param.requires_grad = True
        
        self.loss = nn.CrossEntropyLoss()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def forward(self, input_ids, attention_mask, token_type_ids):
        bert_output = self.bert(input_ids, attention_mask, token_type_ids)
        ## https://huggingface.co/docs/transformers/main_classes/output#transformers.modeling_outputs.BaseModelOutput
        pred = self.output(bert_output.pooler_output)
        pred = self.softmax(pred)
        return pred

    def training_step(self, batch, batch_nb):
        x, t = batch
        y = self.forward(x['input_ids'], x['attention_mask'], x['token_type_ids'])
        loss = self.loss(y, t)
        self.log('train_loss_in_training_step', loss, on_epoch = True)
        print(torch.cuda.get_device_name())
        return {'loss': loss}

    def training_epoch_end(self, training_step_outputs):
        if self.trainer.is_global_zero:
            mean_loss = torch.stack([x['loss'] for x in self.all_gather(training_step_outputs)]).mean()
            self.log("train_loss", float(mean_loss), rank_zero_only=True, logger=True)
    
    def validation_step(self, batch, batch_nb): 
        x, t = batch
        y = self.forward(x['input_ids'], x['attention_mask'], x['token_type_ids'])
        loss = self.loss(y, t)
        self.val_acc(y, t)
        return {'val_loss': loss, 'val_acc': self.val_acc }

    def validation_epoch_end(self, validation_step_outputs):
        #self.log("val_acc", self.val_acc, logger=True, sync_dist=True)        
        if self.trainer.is_global_zero:
            mean_loss = torch.stack([x['val_loss'] for x in self.all_gather(validation_step_outputs)]).mean()
            self.log("val_loss", float(mean_loss), rank_zero_only=True, logger=True)
            self.log("val_acc", self.val_acc, logger=True, on_step = False, on_epoch=True) 
    
    def test_step(self, batch, batch_nb):
        x, t = batch
        y = self.forward(x['input_ids'], x['attention_mask'], x['token_type_ids'])
        self.test_acc(y, t)
        return {'test_acc': self.test_acc }

    def test_epoch_end(self, test_step_outputs):
        #self.log("test_acc", self.test_acc, logger=True, sync_dist=True)
        if self.trainer.is_global_zero:
            self.log("test_acc", self.test_acc, logger=True, on_step = False, on_epoch=True)
    
    def configure_optimizers(self):
        return AdamW([
            {'params': self.bert.encoder.layer[-1].parameters(),
             'lr': self.bert_lr},
            {'params': self.output.parameters(),
             'lr': self.output_lr}
        ])

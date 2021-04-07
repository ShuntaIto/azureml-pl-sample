import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from transformers import BertJapaneseTokenizer
from transformers import BertModel

import pandas as pd
import tarfile
from glob import glob
import linecache
from tqdm import tqdm
import urllib.request

from argparse import ArgumentParser
from azureml_env_adapter import set_environment_variables
from model import BERTClassificationModel
from livedoor_data import LivedoorDatasets

# preprocess


class TokenizerCollate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def collate_fn(self, batch):
        input = [item[0] for item in batch]
        input = self.tokenizer(
            input,
            padding=True,
            max_length=512,
            truncation=True,
            return_tensors="pt")
        targets = torch.tensor([item[1] for item in batch])
        return input["input_ids"], targets

    def __call__(self, batch):
        return self.collate_fn(batch)


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=128, type=int)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    set_environment_variables()

    # ------------
    # data
    # ------------
    dataset = LivedoorDatasets()
    n_samples = len(dataset)
    train_size = int(len(dataset) * 0.8)
    val_size = int(len(dataset) * 0.1)
    test_size = n_samples - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size])

    tokenizer = BertJapaneseTokenizer.from_pretrained(
        'cl-tohoku/bert-base-japanese-whole-word-masking')

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, collate_fn=TokenizerCollate(tokenizer=tokenizer))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            collate_fn=TokenizerCollate(tokenizer=tokenizer))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             collate_fn=TokenizerCollate(tokenizer=tokenizer))

    # ------------
    # model
    # ------------
    model = BERTClassificationModel()

    # fix BERT model
    for param in model.bert.parameters():
        param.requires_grad = False

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_dataloader=train_loader,
                val_dataloaders=val_loader)

    # ------------
    # testing
    # ------------
    result = trainer.test(model, test_dataloaders=test_loader)
    print(result)

    # ------------
    # model saving
    # ------------
    model_path = os.path.join('outputs', 'model.onnx')

    dummy_input = torch.randint(
        low=1, high=10000, size=(10, 512), device='cuda')
    torch.onnx.export(
        model=model,
        args=dummy_input,
        f=model_path,
        opset_version=12,
        verbose=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={'input': {0: 'batch_size', 1: 'text_length'},
                      'output': {0: 'batch_size'}}
    )


if __name__ == "__main__":
    cli_main()

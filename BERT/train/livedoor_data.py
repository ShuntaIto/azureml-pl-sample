import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import pandas as pd
import tarfile
from glob import glob
import linecache
import urllib.request

torch.manual_seed(42)

download_path = "livedoor_news_corpus.tar.gz"
extract_path = "livedoor/"

def download_dataset():
    if not os.path.isfile("livedoor_news_corpus.tar.gz"):
        urllib.request.urlretrieve(
            "https://www.rondhuit.com/download/ldcc-20140209.tar.gz", download_path)
    with tarfile.open(download_path) as t:
        t.extractall(extract_path)

def preprocess():
    ## https://qiita.com/m__k/items/841950a57a0d7ff05506
    categories = [name for name in os.listdir(
        extract_path + "text") if os.path.isdir(extract_path + "text/" + name)]

    datasets = pd.DataFrame(columns=["title", "category"])
    for cat in categories:
        path = extract_path + "text/" + cat + "/*.txt"
        files = glob(path)
        for text_name in files:
            title = linecache.getline(text_name, 3)
            s = pd.Series([title, cat], index=datasets.columns)
            datasets = datasets.append(s, ignore_index=True)

    datasets = datasets.sample(frac=1).reset_index(drop=True)

    categories = list(set(datasets['category']))
    id2cat = dict(zip(list(range(len(categories))), categories))
    cat2id = dict(zip(categories, list(range(len(categories)))))

    datasets['category_id'] = datasets['category'].map(cat2id)
    datasets = datasets[['title', 'category_id']]
    return datasets

class LivedoorDatasets(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        download_dataset()
        datasets = preprocess()
        self.data = list(datasets["title"])
        self.label = list(datasets["category_id"])
        if not len(self.data) == len(self.label):
            raise ValueError("Invalid dataset")
        self.datanum = len(self.data)

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_label = self.label[idx]

        if self.transform:
            out_data = self.transform(["input_ids"][0])

        return out_data, out_label

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
        return input, targets

    def __call__(self, batch):
        return self.collate_fn(batch)

class LivedoorNewsDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, batch_size=32):
        super().__init__()

        self.batch_size = batch_size
        self.tokenizer = tokenizer
        
    def prepare_data(self):
        self.dataset = LivedoorDatasets()

    def setup(self, stage):
        self.n_samples = len(self.dataset)
        self.train_size = int(self.n_samples * 0.8)
        self.val_size = int(self.n_samples * 0.1)
        self.test_size = self.n_samples - self.train_size - self.val_size
        self.train_dataset, self.val_dataset, self.test_dataset = torch.utils.data.random_split(
        self.dataset, [self.train_size, self.val_size, self.test_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=TokenizerCollate(tokenizer=self.tokenizer))
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=TokenizerCollate(tokenizer=self.tokenizer))

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=TokenizerCollate(tokenizer=self.tokenizer))
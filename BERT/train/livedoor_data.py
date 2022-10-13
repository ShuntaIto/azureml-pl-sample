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

download_path = "livedoor_news_corpus.tar.gz"
extract_path = "livedoor/"

def download_dataset():
    if not os.path.isfile("livedoor_news_corpus.tar.gz"):
        urllib.request.urlretrieve(
            "https://www.rondhuit.com/download/ldcc-20140209.tar.gz", download_path)
    with tarfile.open(download_path) as t:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(t, extract_path)

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

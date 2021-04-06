import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from tqdm import tqdm
import torchvision.models as models
from argparse import ArgumentParser
from model import ResNetClassificationModel

def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=128, type=int)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
    
    dataset = CIFAR10(".", train=True, download=True,transform=transform)
    n_samples = len(dataset) 
    train_size = int(len(dataset) * 0.9) 
    val_size = n_samples - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    test_dataset = CIFAR10(".", train=False, download=True, transform=transform)
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # ------------
    # model
    # ------------
    model = ResNetClassificationModel()
    #for param in model.resnet18.parameters():
    #    param.requires_grad = False

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader) 
    
    # ------------
    # testing
    # ------------
    result = trainer.test(model,test_dataloaders=test_loader)
    print(result)

    # ------------
    # model saving
    # ------------
    model_path = os.path.join('outputs', 'model.onnx')
    
    dummy_input = torch.randn(10, 3, 224, 224, device='cuda')
    torch.onnx.export(
        model=model,
        args=dummy_input,
        f=model_path,
        opset_version=12,
        verbose=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}}
    )

if __name__ == "__main__":
    cli_main()

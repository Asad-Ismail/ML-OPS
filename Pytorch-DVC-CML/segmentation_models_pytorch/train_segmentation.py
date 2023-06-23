import os
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from pprint import pprint
from torch.utils.data import DataLoader
from segmentation_model import PetModel
from pytorch_lightning.callbacks import ModelCheckpoint
import yaml
from dvclive import Live


# Load parameters from yaml file
with open('params.yaml') as file:
    params = yaml.safe_load(file)

data_dir="pets_data"
from segmentation_models_pytorch.datasets import SimpleOxfordPetDataset


# init train, val, test sets
train_dataset = SimpleOxfordPetDataset(data_dir, "train")
valid_dataset = SimpleOxfordPetDataset(data_dir, "valid")
test_dataset = SimpleOxfordPetDataset(data_dir, "test")

# It is a good practice to check datasets don`t intersects with each other
assert set(test_dataset.filenames).isdisjoint(set(train_dataset.filenames))
assert set(test_dataset.filenames).isdisjoint(set(valid_dataset.filenames))
assert set(train_dataset.filenames).isdisjoint(set(valid_dataset.filenames))

print(f"Train size: {len(train_dataset)}")
print(f"Valid size: {len(valid_dataset)}")
print(f"Test size: {len(test_dataset)}")

n_cpu = os.cpu_count()
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=n_cpu)
valid_dataloader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=n_cpu)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=n_cpu)

print(f"Building Model!!")

checkpoint_dir = "modelCheckpoints"
checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir,
    monitor='valid_dataset_iou',
    filename='model-best',
    save_top_k=1,
    mode='max')

with Live(save_dvc_exp=False) as live:
    
    model = PetModel("FPN", "resnet34", in_channels=3, out_classes=1,dvc=live)

    trainer = pl.Trainer(
        accelerator="auto",
        callbacks=[checkpoint_callback],
        max_epochs=params['train']['epochs'] # specify the number of epochs you want to train
    )

    # Fit the model
    trainer.fit(model,
    train_dataloaders=train_dataloader, 
    val_dataloaders=valid_dataloader)
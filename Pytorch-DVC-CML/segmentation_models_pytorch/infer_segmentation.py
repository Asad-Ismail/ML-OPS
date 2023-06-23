import os
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from pprint import pprint
from torch.utils.data import DataLoader
from segmentation_model import PetModel
import cv2
import numpy as np

data_dir="pets_data"
from segmentation_models_pytorch.datasets import SimpleOxfordPetDataset


# init train, val, test sets
valid_dataset = SimpleOxfordPetDataset(data_dir, "valid")
test_dataset = SimpleOxfordPetDataset(data_dir, "test")

print(f"Valid size: {len(valid_dataset)}")

valid_dataloader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=1)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=1)

#print(f"Building Model!!")

model = PetModel("FPN", "resnet34", in_channels=3, out_classes=1)

print(f"Load checkpoint of model!")
# Path to your checkpoint
checkpoint_path = "lightning_logs/version_0/checkpoints/epoch=4-step=1035.ckpt"

# Load the model from the checkpoint
model.load_state_dict(torch.load(checkpoint_path)["state_dict"])

trainer = pl.Trainer()

# If we want to test the model
#test_metrics = trainer.test(model, dataloaders=test_dataloader, verbose=False)
#pprint(test_metrics)

batch = next(iter(test_dataloader))
with torch.no_grad():
    model.eval()
    logits = model(batch["image"])
pr_masks = logits.sigmoid()

# Define a color map for masks
color_map = [[0, 0, 128], [0, 128, 0], [128, 0, 0]]  # Blue, Green, Red

for i, (image, gt_mask, pr_mask) in enumerate(zip(batch["image"], batch["mask"], pr_masks)):
    image = image.permute(1, 2, 0).numpy().astype('uint8')[...,::-1]
    gt_mask = gt_mask.numpy().squeeze().astype('uint8')
    pr_mask = pr_mask.numpy().squeeze()
    pr_mask= np.where(pr_mask<0.5,0,1).astype('uint8')

    # Convert masks to RGB and add a color map
    gt_mask_rgb = cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2BGR)
    pr_mask_rgb = cv2.cvtColor(pr_mask, cv2.COLOR_GRAY2BGR)

    for c in range(3):
        gt_mask_rgb[:, :, c] = np.where(gt_mask_rgb[:, :, c] > 0, color_map[1][c], 0)
        pr_mask_rgb[:, :, c] = np.where(pr_mask_rgb[:, :, c] > 0, color_map[2][c], 0)

    # Overlay masks on the original image
    overlay_gt = cv2.addWeighted(image, 0.7, gt_mask_rgb, 0.3, 0)
    overlay_pr = cv2.addWeighted(image, 0.7, pr_mask_rgb, 0.3, 0)

    # Concatenate and save the image
    concat_image = cv2.hconcat([overlay_gt, overlay_pr])
    cv2.imwrite(f"vis_results/output_{i}.png", concat_image)


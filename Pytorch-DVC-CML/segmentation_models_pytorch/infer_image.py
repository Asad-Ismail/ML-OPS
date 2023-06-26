from flask import Flask, request, send_file
from PIL import Image
import io
import torch
import segmentation_models_pytorch as smp
import numpy as np
import cv2
from segmentation_model import PetModel

# Load your trained model
model = PetModel("FPN", "resnet34", in_channels=3, out_classes=1)
checkpoint_path = "modelCheckpoints/model-best.ckpt"
model.load_state_dict(torch.load(checkpoint_path)["state_dict"])

img_path="pets_data/images/english_cocker_spaniel_141.jpg"

image = np.array(Image.open(img_path).convert('RGB').resize((256, 256), Image.BILINEAR))
vis_img=image.copy()
image=np.moveaxis(image, -1, 0)

# Define a color map for masks
color_map = [[0, 0, 128], [0, 128, 0], [128, 0, 0]]  # Blue, Green, Red

# Add batch dimension and send image through your model
model.eval()
with torch.no_grad():
    input_tensor = torch.from_numpy(image).unsqueeze(0)
    logits = model(input_tensor)
pr_masks = logits.sigmoid()

# Process the output as you did in your original code
pr_mask = pr_masks[0].numpy().squeeze()
pr_mask = np.where(pr_mask<0.5,0,1).astype('uint8')

# Convert masks to RGB and add a color map
pr_mask_rgb = cv2.cvtColor(pr_mask, cv2.COLOR_GRAY2BGR)

for c in range(3):
    pr_mask_rgb[:, :, c] = np.where(pr_mask_rgb[:, :, c] > 0, color_map[2][c], 0)

# Overlay masks on the original image
overlay_pr = cv2.addWeighted(vis_img, 0.4, pr_mask_rgb, 0.6, 0)

cv2.imwrite(f"test.png",overlay_pr)

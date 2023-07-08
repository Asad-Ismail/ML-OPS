from minio import Minio
import numpy as np
import os

# Connect to MinIO
minio_client = Minio(
    "localhost:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)

# Get list of entropy files in MinIO
objects = minio_client.list_objects("entropies")

# Download entropy files and get corresponding image names
entropy_image_pairs = []
for obj in objects:
    data = minio_client.get_object("entropies", obj.object_name)
    entropy = np.load(data)
    image_name = obj.object_name.replace(".npy", ".jpg")  # Assumes image names match entropy names
    entropy_image_pairs.append((entropy, image_name))

# Sort images by entropy
sorted_images = sorted(entropy_image_pairs, key=lambda x: x[0], reverse=True)

# Select top N images for annotation
N = 10
selected_images = sorted_images[:N]

# Send images for annotation (this depends on how you do your annotations)
for entropy, image_name in selected_images:
    image_data = minio_client.get_object("images", image_name)
    send_for_annotation(image_data)  # You'd need to implement this

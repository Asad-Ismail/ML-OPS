from minio import Minio
import numpy as np
import os
import requests

# Connect to MinIO
minio_client = Minio(
    "localhost:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)


def send_for_annotation(image_data, image_name):
    # Your CVAT instance url
    CVAT_URL = "http://localhost:8080"

    # Your CVAT credentials
    auth = ("admin", "password")

    # Define a task
    task_data = {
        "name": "My annotation task",
        "labels": [{"name": "object"}]
    }

    # Create a new task
    response = requests.post(f"{CVAT_URL}/api/v1/tasks", json=task_data, auth=auth)
    if response.status_code != 201:
        print(f"Failed to create task: {response.text}")
        return

    # Get the ID of the created task
    task_id = response.json()["id"]

    # Upload the image to the task
    response = requests.post(
        f"{CVAT_URL}/api/v1/tasks/{task_id}/data",
        files={"client_files[0]": (image_name, image_data)},
        auth=auth
    )
    if response.status_code != 202:
        print(f"Failed to upload image: {response.text}")
        return

    print(f"Successfully sent image {image_name} for annotation")


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

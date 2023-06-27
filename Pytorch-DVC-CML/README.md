# Image Semantic Segmentation Pipeline

This project consists of a semantic segmentation machine learning pipeline for pet images. We've implemented a Flask server to serve the segmentation model and leveraged Prometheus and Grafana for monitoring, DVC for data versioning, and GitHub Actions and CML for automated training and deployment.

Below is the high-level architecture of our pipeline.

## Installation


### Install promethus Grafana and minio

```
brew install prometheus grafana

```

### Run Prometheus

```
prometheus --config.file=prometheus.yml

```

### Run Grafana

brew services start grafana

### Install minio

brew install minio/stable/minio

### Run minio Server

minio server /home/shared

Grafana should be running in the background. You can access the Grafana dashboard by opening your web browser and navigating to http://localhost:3000. The default username and password are both 'admin'.

You can test the deployment using the MinIO Console, an embedded web-based object browser built into MinIO Server. Point a web browser running on the host machine to http://127.0.0.1:9000 and log in with the root credentials. The MinIO deployment starts using default root credentials minioadmin:minioadmin



Pipeline Architecture:

## Image Inference Server:
At the heart of the pipeline is the image inference server, implemented in Flask. The server takes in pet images and passes them through a pre-trained semantic segmentation model. It returns the prediction as a response.
## Model Monitoring:
To monitor the performance of our model, we're using Prometheus and Grafana. The server exports metrics like prediction counts, processing times, and confidence scores to Prometheus. Grafana is used to visualize these metrics and set up alerts.
## Data Versioning and Training:
We're using Data Version Control (DVC) for dataset versioning and tracking changes. Whenever the alert from Grafana triggers indicating that the model needs to be retrained, we make use of GitHub Actions and CML to start the retraining pipeline.
## Active Learning and Data Annotation:
The system uses active learning principles to improve the training data over time. Images with higher entropy, indicating that the model is uncertain about its prediction, are flagged for annotation.
## Data Storage:
We store input images and their corresponding entropy scores in MinIO, an open-source object storage service, for further curation and annotation.


## Step-by-step Guide

# Setup:
Install all necessary tools and clone the GitHub repository. Start your Flask server and Prometheus server. Configure Prometheus to scrape metrics from your Flask server. Start Grafana and configure it to visualize the metrics from Prometheus.
# Inference:
Send images to the Flask server using POST requests to the /predict endpoint. The server processes the image and returns a semantic segmentation mask.
# Monitoring:
The Flask server exports metrics to Prometheus. Grafana visualizes these metrics and sets up alerts based on thresholds or changes in model performance.
# Active Learning:
When Grafana triggers an alert, indicating that the model's performance has dropped, the Flask server starts exporting the entropy of model predictions along with the corresponding input images to MinIO.
# Data Annotation:
The stored images are then retrieved, sorted by entropy, and annotated. High entropy images, where the model is unsure about its predictions, are given priority for annotation.
# Retraining:
Once the new annotations are ready, DVC tracks the changes to the dataset. The updated dataset triggers GitHub Actions to retrain the model.
# Continuous Monitoring and Learning:
This process continues in a loop, where the model's performance is continuously monitored. Any drop in performance triggers active learning, leading to retraining and re-deployment of the model.
#This pipeline ensures continuous improvement in model performance by using active learning principles and regular monitoring of model performance. It also automates the process of retraining and deploying the model using DVC, GitHub Actions, and CML. With this pipeline, we're able to maintain a high-performance semantic segmentation model for pet images.


from flask import Flask, request, send_file
from flask import Flask, send_from_directory, request, jsonify, Response
from PIL import Image
import io
import torch
import segmentation_models_pytorch as smp
import numpy as np
import cv2
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter, Histogram, Summary
from segmentation_model import PetModel
import time

# Load your trained model
model = PetModel("FPN", "resnet34", in_channels=3, out_classes=1)
checkpoint_path = "modelCheckpoints/model-best.ckpt"
model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu'))["state_dict"])

# Define a color map for masks
color_map = [[0, 0, 128], [0, 128, 0], [128, 0, 0]]  # Blue, Green, Red

app = Flask(__name__)

metrics = PrometheusMetrics(app)

# Metrics
predict_counter = Counter('predictions', 'The total number of predictions')
prediction_scores = Summary('prediction_scores', 'Quantiles of prediction scores', ['Quant'])
non_zero_confidence = Summary('non_zero_confidence', 'Average confidence for non-zero predictions')
model_summary = Summary('processing_time_model', 'Time spend forwad time model')
predict_summary = Summary('processing_time_overall', 'Time spend processing request')


@app.route('/predict', methods=['POST'])
@predict_summary.time()
def predict():
    # Get image from the POST request
    if 'file' not in request.files:
        return 'No file part', 400

    file = request.files['file'].read()  # read the file bytes
    image = Image.open(io.BytesIO(file))  # open the image
    image = np.array(image.resize((256, 256), Image.LINEAR)) 

    vis_image= image.copy()
    image=np.moveaxis(image, -1, 0)

    # Add batch dimension and send image through your model
    model.eval()
    with torch.no_grad():
        input_tensor = torch.from_numpy(image).unsqueeze(0)
        start_time = time.monotonic()
        logits = model(input_tensor)
        # End time
        end_time = time.monotonic()
        processing_time = end_time - start_time
        model_summary.observe(processing_time)

    pr_masks = logits.sigmoid()

    # Process the output as you did in your original code
    pr_mask = pr_masks[0].numpy().squeeze()

    for quant in [0.1,0.2,0.3,0.4,0.5,0.6, 0.70, 0.80, 0.9, 0.99]:  # choose the quantiles you are interested in
        score = np.percentile(pr_mask.copy().flatten(), quant * 100)
        prediction_scores.labels(Quant=quant).observe(score)

    # Compute the average confidence for non-zero predictions
    non_zero_scores = pr_mask.copy().flatten()[pr_mask.copy().flatten() > 1e-2]
    non_zero_average = np.mean(non_zero_scores)
    non_zero_confidence.observe(non_zero_average)

    pr_mask = np.where(pr_mask<0.5,0,1).astype('uint8')

    # Convert masks to RGB and add a color map
    pr_mask_rgb = cv2.cvtColor(pr_mask, cv2.COLOR_GRAY2BGR)
    for c in range(3):
        pr_mask_rgb[:, :, c] = np.where(pr_mask_rgb[:, :, c] > 0, color_map[2][c], 0)

    # Overlay masks on the original image
    overlay_pr = cv2.addWeighted(vis_image, 0.3, pr_mask_rgb, 0.7, 0)

    # Save the result to a BytesIO object
    result = Image.fromarray(overlay_pr.astype('uint8'))
    byte_arr = io.BytesIO()
    result.save(byte_arr, format='PNG')
    byte_arr.seek(0)

    # Increment counter
    predict_counter.inc()

    # Send result image
    return send_file(byte_arr, mimetype='image/png')


# Add route for metrics
@app.route('/metrics')
def metrics():
    return Response(prometheus_client.generate_latest(), mimetype=CONTENT_TYPE_LATEST)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

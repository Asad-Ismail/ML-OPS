import requests
from PIL import Image
import io

# Open image and convert it to RGB

img_path="pets_data/images/english_cocker_spaniel_141.jpg"

image = Image.open(img_path).convert('RGB')

# Save image to a BytesIO object
byte_arr = io.BytesIO()
image.save(byte_arr, format='PNG')
byte_arr.seek(0)

# Send image data to the server
response = requests.post('http://localhost:5000/predict', files={'file': byte_arr})

# Make sure the request was successful
response.raise_for_status()

# Read the processed image from the response and save it to a file
result_image = Image.open(io.BytesIO(response.content))
result_image.save('output_image.png')

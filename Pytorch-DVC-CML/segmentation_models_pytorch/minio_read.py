from minio import Minio
from minio.error import S3Error
from minio.deleteobjects import DeleteObject
import numpy as np
import io


# Create a Minio client
minio_client = Minio(
    "localhost:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)


def load_data_from_minio(bucket_name):
    # Initialize an empty dictionary to hold the data
    data_dict = {}

    # List all objects in the bucket
    objects = minio_client.list_objects(bucket_name)

    # Iterate over the objects
    for obj in objects:
        # Use the object name as the key
        key = obj.object_name

        print(f"Key is {key}")

        # Get the object data
        data = minio_client.get_object(bucket_name, key)

        # The data is returned as a stream, so we need to read it into memory
        data_bytes = data.read()

        # Convert the data from bytes back into a numpy array
        data_array = np.load(io.BytesIO(data_bytes))

        # Add the data to the dictionary
        data_dict[key] = data_array

    return data_dict



def clear_bucket(bucket_name):
    # Get a list of all objects in the bucket
    objects = minio_client.list_objects(bucket_name)

    # Convert the list of objects to a list of DeleteObject instances
    object_names = [DeleteObject(obj.object_name) for obj in objects]

    # Delete all objects
    errors = minio_client.remove_objects(bucket_name, object_names)

    # The remove_objects function returns a generator that yields errors, if any occurred
    for error in errors:
        print(f"Error: {error}")

# Use the function to clear the 'images' and 'entropies' buckets
#clear_bucket('images')
#clear_bucket('entropies')




# Use the function to load the data from the 'images' and 'entropies' buckets
images_dict = load_data_from_minio('images')
entropies_dict = load_data_from_minio('entropies')

print(entropies_dict.keys())
print(images_dict.keys())

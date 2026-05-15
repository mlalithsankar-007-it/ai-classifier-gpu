import os
import shutil
import boto3
import tensorflow as tf
import numpy as np

from PIL import Image


MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")

MODEL_BUCKET = os.getenv("MODEL_BUCKET", "models")
MODEL_OBJECT = os.getenv(
    "MODEL_OBJECT",
    "classifier/v1/interior_exterior_classifier.h5"
)

LOCAL_MODEL_PATH = "/tmp/model.h5"


def download_model():

    if os.path.exists(LOCAL_MODEL_PATH):
        return

    s3 = boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY
    )

    print("Downloading model from MinIO...")

    s3.download_file(
        MODEL_BUCKET,
        MODEL_OBJECT,
        LOCAL_MODEL_PATH
    )

    print("Model downloaded successfully")


download_model()

model = tf.keras.models.load_model(LOCAL_MODEL_PATH)


def classify_and_organize(files, output_dir):

    interior_dir = os.path.join(output_dir, "interior")
    exterior_dir = os.path.join(output_dir, "exterior")

    os.makedirs(interior_dir, exist_ok=True)
    os.makedirs(exterior_dir, exist_ok=True)

    for file_path in files:

        try:

            img = Image.open(file_path).convert("RGB")

            img = img.resize((224, 224))

            img_array = np.array(img) / 255.0

            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)[0][0]

            filename = os.path.basename(file_path)

            if prediction > 0.5:

                shutil.copy(
                    file_path,
                    os.path.join(interior_dir, filename)
                )

            else:

                shutil.copy(
                    file_path,
                    os.path.join(exterior_dir, filename)
                )

        except Exception as e:

            print(f"Error processing {file_path}: {e}")
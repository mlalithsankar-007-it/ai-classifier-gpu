import os
import shutil
import traceback
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
USE_GPU = os.getenv("USE_GPU", "true").lower() in {"1", "true", "yes"}

MODEL = None


def _configure_environment():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    if not USE_GPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("GPU disabled manually.")
        return

    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        print(f"Detected GPUs: {gpus}")

        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            print("TensorFlow GPU memory growth enabled.")

        except Exception as e:
            print(f"Error configuring GPU memory growth: {e}")

    else:
        print("No GPU detected. Falling back to CPU.")


def download_model():
    if os.path.exists(LOCAL_MODEL_PATH):
        print("Model already exists locally.")
        return

    if not MINIO_ENDPOINT or not MINIO_ACCESS_KEY or not MINIO_SECRET_KEY:
        raise RuntimeError(
            "Missing MinIO credentials: MINIO_ENDPOINT, MINIO_ACCESS_KEY, and MINIO_SECRET_KEY must be set."
        )

    print("Connecting to MinIO...")

    s3 = boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY
    )

    print("Downloading model from MinIO...")
    s3.download_file(MODEL_BUCKET, MODEL_OBJECT, LOCAL_MODEL_PATH)
    print("Model downloaded successfully.")


class LegacyInputLayer(tf.keras.layers.InputLayer):
    def __init__(self, *args, batch_shape=None, **kwargs):
        if batch_shape is not None:
            kwargs["batch_input_shape"] = batch_shape
        super().__init__(*args, **kwargs)


def load_model_if_needed():
    global MODEL

    if MODEL is not None:
        return MODEL

    _configure_environment()

    download_model()

    print("Loading TensorFlow model...")

    MODEL = tf.keras.models.load_model(
        LOCAL_MODEL_PATH,
        compile=False,
        custom_objects={
            "InputLayer": LegacyInputLayer
        }
    )

    print("Model loaded successfully.")

    return MODEL


def classify_and_organize(files, output_dir):
    model = load_model_if_needed()

    interior_dir = os.path.join(output_dir, "interior")
    exterior_dir = os.path.join(output_dir, "exterior")

    os.makedirs(interior_dir, exist_ok=True)
    os.makedirs(exterior_dir, exist_ok=True)

    print(f"Classifying {len(files)} files...")

    for file_path in files:
        try:
            img = Image.open(file_path).convert("RGB")

            img = img.resize((224, 224))

            img_array = np.array(img, dtype=np.float32) / 255.0

            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array, verbose=0)

            prediction_value = float(prediction[0][0])

            filename = os.path.basename(file_path)

            print(
                f"{filename} -> prediction={prediction_value}"
            )

            target_dir = (
                interior_dir
                if prediction_value > 0.5
                else exterior_dir
            )

            shutil.copy(
                file_path,
                os.path.join(target_dir, filename)
            )

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

            traceback.print_exc()
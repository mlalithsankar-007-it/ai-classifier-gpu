import tensorflow as tf
import numpy as np
from PIL import Image
import os
import shutil

MODEL_PATH = "app/model/interior_exterior_classifier.h5"

model = tf.keras.models.load_model(MODEL_PATH)


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
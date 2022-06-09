import os
import onnxruntime
import numpy as np
import urllib.request
import json
from PIL import Image, ImageDraw
import datetime
from datetime import timezone

import logging

IMG_WIDTH = 512
IMG_HEIGHT = 344


def softmax(x):
    """Compute softmax values (probabilities from 0 to 1) for each possible label."""
    x = x.reshape(-1)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def predict(model):
    model = os.path.join("assets", model)
    if not model:
        logging.error("Could not find ONNX model file")
        return json.dumps({'error': "No model specified"})

    labels_file = "./assets/classes.json"
    with open(labels_file) as f:
        classes = json.load(f)

    # Download latest image
    urllib.request.urlretrieve("http://tristarobservatory.com/obscam/allsky/image.png", ".\\tmp.png")
    image_path = ".\\tmp.png"

    # Resize
    img = Image.open(image_path)
    img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.LANCZOS)

    # Remove burned in information
    draw = ImageDraw.Draw(img)
    draw.rectangle(((0, 0), (120, 85)), fill="black")

    # save it as temp file
    img.save(image_path)

    # Get UTC time stamp
    dt = datetime.datetime.now(timezone.utc)
    utc_time = dt.replace(tzinfo=timezone.utc)
    utc_timestamp = utc_time.timestamp()

    reference_image = np.array(Image.open(image_path), dtype=np.float32)
    reference_image = reference_image.transpose()
    reference_image = np.expand_dims(reference_image, axis=2)

    # ONNX Run Inference
    session = onnxruntime.InferenceSession(model, None)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    result = session.run([output_name], {input_name: [reference_image]})
    prediction = int(np.argmax(np.array(result).squeeze(), axis=0))
    score = softmax(result[0])[prediction]

    return json.dumps({'allSkyAIClass': classes[prediction], 'allSkyAIConfidence': 100 * score, 'utc': utc_timestamp})

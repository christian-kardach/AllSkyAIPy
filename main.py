import os
import uuid
from flask import Flask, request
from flask_cors import CORS, cross_origin
import json

from allSkyAI import predict

app = Flask(__name__)
cors = CORS(app)


@app.route('/<string:name>/', methods=['GET'])
@cross_origin()
def predict_image(name):
    if name == "tristar":
        model = "tristar.onnx"
        return predict(model)
    else:
        return json.dumps({'error': 'no such AllSky here...'})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
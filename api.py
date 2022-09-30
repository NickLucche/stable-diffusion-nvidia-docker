from flask import Flask
from traitlets import List
# from main import inference, MP
from flask import request, send_from_directory
import os
import io
from PIL import Image
from base64 import encodebytes
import numpy as np
# from flask_cors import CORS, cross_origin


app = Flask(__name__, static_folder='stable-diffusion-ui/build')
# cors = CORS(app)
# app.config['CORS_HEADERS'] = 'Content-Type'

def validate_request()->bool:
    return True

@app.route("/image_generation", methods=['POST'])
def image_gen():
    # if not request.is_json or not validate_request(request.json):
        # return "", 400
    # run inference
    # images:List[Image.Image] = inference(**request.json)
    rnd = (np.random.randn(32, 32, 3)*255).astype(np.uint8)
    images = [Image.fromarray(rnd) for _ in range(1)]

    # encode each image
    encoded = []
    for im in images:
        byte_arr = io.BytesIO()
        # convert the PIL image to byte array
        im.save(byte_arr, format='PNG') 
        # encode as base64
        encoded.append(encodebytes(byte_arr.getvalue()).decode('ascii'))

    return {"images": encoded}

# Serve React App, from https://stackoverflow.com/a/45634550/4991653
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')


if __name__ == "__main__":
    # use_reloader=True?
    app.run()

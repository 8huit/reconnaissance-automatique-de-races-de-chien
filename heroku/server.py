import os
import uvicorn
import logging
from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import tensorflow as tf
import joblib
import numpy as np
import pandas as pd
import io
from PIL import Image
import cv2
import base64

# Initialize logging
#my_logger = logging.getLogger()
#my_logger.setLevel(logging.DEBUG)
#logging.basicConfig(level=logging.DEBUG, filename='logs.log')

app = FastAPI()

app.mount("/bootstrap-5.1.3-dist/css", StaticFiles(directory="templates/bootstrap-5.1.3-dist/css"), name="css")
app.mount("/bootstrap-5.1.3-dist/js", StaticFiles(directory="templates/bootstrap-5.1.3-dist/js"), name="js")
app.mount("/custom/js", StaticFiles(directory="templates/custom/js"), name="cjs")

templates = Jinja2Templates(directory="templates")

@app.on_event("startup")
def load_model():
    global toto
    global model
    global image
    global dictDogTypes

    toto="toutou"
    model = tf.keras.models.load_model("./models/cnn_vgg19")
    dictDogTypes = joblib.load("./models/dictDogTypes.jbl.bz2")

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/")
async def home(request: Request, upload: UploadFile = File(...)):
    # get uploaded image properties
    filename=upload.filename
    contents = await upload.read()
    buf = io.BytesIO(contents)
    image_array=np.array(Image.open(buf)) # numpy array conversion
    filesize=buf.getbuffer().nbytes
    
    #resize for prediction
    image_array=cv2.resize(image_array, (256, 256),interpolation = cv2.INTER_NEAREST)
    #prepare image (add dimension) for model
    image_array=(np.expand_dims(image_array, axis=0)).astype(float)
    
    # predict dog class
    x_pred=model.predict(image_array)
    y_pred=np.argmax(x_pred[0])
    result=dictDogTypes[y_pred]

    # encode base64 to send back to image uri
    encoded = base64.b64encode(contents).decode("utf-8")
    datauri="data:image/jpg;base64,"+encoded

    identifytext="Le chien sur l'image "+filename+" ("+str(filesize)+" octets) correspond à la famille des "+result
    return templates.TemplateResponse("index.html", {"request": request,"identifytext":identifytext,"uploadedimg":datauri})

# Run the API with uvicorn
envPORT=os.environ.get('PORT','8000')
if __name__ == '__main__':
   uvicorn.run(app, host='0.0.0.0', port=int(envPORT))

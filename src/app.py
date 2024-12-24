"""Simple API to inference the Model"""
from fastapi import FastAPI

from loguru import logger
import api.schemas as schemas
from api.constants import MODEL_PATH
import tensorflow as tf
import modelling.architecture as architecture
import numpy as np

import os
# Run on CPU on the container
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# Init the api and model
app = FastAPI()
model = None

def load_model() -> None:
    """Loads the SMM model."""
    logger.info("Loading the SMM model")
    global model
    model = tf.keras.models.load_model(MODEL_PATH,
        custom_objects={'contrastive_loss': architecture.contrastive_loss,
                         'CosineSimilarityLayer': architecture.CosineSimilarityLayer})
    logger.success("Model loaded Succesfully!")

# Load the model at startup
@app.on_event("startup")
async def startup_event():
    load_model()


@app.get("/status")
async def get_status():
    return {"status": "OK"}


@app.get("/model_summary")
async def get_model_summary():
    """Get Model Summary"""
    global model
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)
    return {"model_summary": short_model_summary}


@app.post("/inference")
async def inference(request: schemas.InferenceRequest):
    """Endpoint for single inference of two tracks."""
    global model
    logger.info("Starting inference...")
    # Reshape
    inputa = np.array(request.input_data[0])
    inputa = np.expand_dims(inputa, axis=0)  
    inputb = np.array(request.input_data[1])
    inputb = np.expand_dims(inputb, axis=0)  

    preds = model.predict([inputa, inputb]).tolist()[0][0]
    logger.info("Inference completed")
    
    return {"Similarity score": preds}

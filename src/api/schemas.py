"""Pydantic model for inference request."""
from pydantic import BaseModel

class InferenceRequest(BaseModel):
    input_data: list
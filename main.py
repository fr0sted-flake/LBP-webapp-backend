from fastapi import FastAPI
from pydantic import BaseModel
from model.predict import predict_all

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

class InputParams(BaseModel):
    throttle_pos: float
    gear: float

@app.post("/predict")
def predict(params: InputParams):
    result = predict_all(params.throttle_pos, params.gear)
    return result

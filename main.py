from fastapi import FastAPI
from pydantic import BaseModel
from model.predict import predict_all
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow all origins for development (or specify your frontend port like "http://localhost:3000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to ["http://localhost:3000"] in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows GET, POST, OPTIONS, etc.
    allow_headers=["*"],
)

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

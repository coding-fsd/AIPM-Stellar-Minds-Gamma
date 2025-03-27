from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import ollama

app = FastAPI()

# Allow frontend to access the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load cleaned CSV file directly
def load_data():
    return pd.read_csv("cleaned_data.csv")

@app.get("/data")
def get_data():
    """Returns the dataset to the frontend"""
    df = load_data()
    return df.to_dict(orient="records")

@app.post("/label")
def label_data():
    """Labels the dataset using TinyLlama"""
    df = load_data()
    prompt = f"Label this dataset based on sensor readings:\n{df.to_json()}"

    response = ollama.chat(model="tinyllama", messages=[{"role": "user", "content": prompt}])
    
    return {"labels": response['message']['content']}

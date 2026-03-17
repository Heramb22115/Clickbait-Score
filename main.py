from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import joblib

print("Waking up the AI model...")
try:
    model = joblib.load("models/clickbait_model.pkl")
except FileNotFoundError:
    print("ERROR: Model not found. Did you run train.py first?")
    exit()

app = FastAPI(title="Robust Clickbait API")

# ROBUSTNESS: Require a string that is at least 3 characters long
class TextRequest(BaseModel):
    text: str = Field(..., min_length=3, description="The headline to analyze. Must be at least 3 characters.")

@app.post("/score")
async def score_headline(request: TextRequest):
    try:
        # Clean up any accidental extra spaces from the user
        headline = request.text.strip()
        
        # Get predictions
        probabilities = model.predict_proba([headline])[0]
        clickbait_score = round(probabilities[1] * 100, 2)
        is_clickbait = bool(clickbait_score > 50.0)
        
        return {
            "headline": headline,
            "clickbait_probability_score": clickbait_score,
            "is_clickbait": is_clickbait,
            "status": "success"
        }
    except Exception as e:
        # ROBUSTNESS: If anything fails, return a clean 500 error instead of crashing
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")

# This route now serves our HTML UI instead of just a JSON message
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()
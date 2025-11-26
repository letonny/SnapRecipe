import os
import cv2
import numpy as np
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from ultralytics import YOLO

# gemini
import google.generativeai as genai
from dotenv import load_dotenv

# load .env file
load_dotenv()

# API key
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_KEY:
    print("⚠️ WARNING: GEMINI_API_KEY not found in .env file. Recipe generation will fail.")

genai.configure(api_key=GEMINI_KEY)

# two models: 'gemini-1.5-flash' is faster / 'gemini-1.5-pro' is longer
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

app = FastAPI(title="SnapRecipe Backend")

# using YOLO
model = YOLO("yolov8n.pt") 
KNOWN_FOOD_CLASSES = {
    'apple', 'banana', 'orange', 'broccoli', 'carrot', 'cake', 'sandwich', 
    'donut', 'pizza', 'hot dog', 'bottle', 'bowl', 'cup'
}

# data models
# use Pydantic for validation / tell Gemini what structure to return
class RecipeStep(BaseModel):
    step_number: int
    instruction: str

class RecipeResponse(BaseModel):
    title: str
    cook_time: str
    difficulty: str
    ingredients_used: List[str]
    missing_ingredients: List[str]
    steps: List[RecipeStep]
    chef_note: str

class RecipeRequest(BaseModel):
    ingredients: List[str]
    difficulty: str  
    dietary_restrictions: Optional[str] = None 

# endpoint

@app.post("/detect")
async def detect_ingredients(file: UploadFile = File(...)):
    """
    Phase 1: Detect ingredients (unchanged)
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        results = model(img)
        detected_items = set()
        
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                if class_name in KNOWN_FOOD_CLASSES:
                    detected_items.add(class_name)

        return {
            "message": "Detection successful",
            "detected_count": len(detected_items),
            "ingredients": list(detected_items)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-recipe", response_model=RecipeResponse)
async def generate_recipe(request: RecipeRequest):
    """
    Phase 2: Generate Recipe using Google Gemini (Async Version)
    """
    if not GEMINI_KEY:
        raise HTTPException(status_code=500, detail="Server misconfigured: Missing Gemini API Key")

    prompt = f"""
    You are SnapRecipe, an expert AI chef.
    Create a unique recipe based on these parameters:
    
    - Available Ingredients: {', '.join(request.ingredients)}
    - Skill Level: {request.difficulty}
    - Dietary Restrictions: {request.dietary_restrictions if request.dietary_restrictions else "None"}
    
    Requirements:
    1. Use as many available ingredients as possible.
    2. Assume basic pantry staples (oil, salt, pepper, water) are available.
    3. If crucial ingredients are missing, list them in 'missing_ingredients'.
    4. Keep steps clear and actionable.
    """

    try:
        # ✅ FIX: Use generate_content_async with await
        response = await gemini_model.generate_content_async(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=RecipeResponse
            )
        )
        
        # Parse the JSON text from Gemini
        recipe_data = json.loads(response.text)
        return recipe_data

    except Exception as e:
        print(f"Gemini Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate recipe. AI might be busy.")

@app.get("/")
def root():
    return {"message": "SnapRecipe Backend with Gemini is running!"}
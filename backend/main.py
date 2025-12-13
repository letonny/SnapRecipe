import os
import cv2
import numpy as np
import json
import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from ultralytics import YOLO
import google.generativeai as genai
from dotenv import load_dotenv

# the setup
load_dotenv()

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_KEY:
    print("⚠️ WARNING: GEMINI_API_KEY not found. Recipe generation will fail.")
else:
    genai.configure(api_key=GEMINI_KEY)

# must not change
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB limit
ALLOWED_CONTENT_TYPES = ["image/jpeg", "image/png", "image/webp"]
KNOWN_FOOD_CLASSES = {
    'apple', 'banana', 'orange', 'broccoli', 'carrot', 'cake', 'sandwich', 
    'donut', 'pizza', 'hot dog', 'bottle', 'bowl', 'cup'
}

# Initialize Models (Load once at startup)
# 'yolov8n.pt' is small, or upgrade to 'yolov8x', loading might take time.
yolo_model = YOLO("yolov8n.pt") 
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

app = FastAPI(title="SnapRecipe Backend")

# middleware
# connect to frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace "*" with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DATA MODELS ---
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
    cuisine: Optional[str] = "Any"  # <--- NEW FIELD (Default: "Any")

# functions
def run_yolo_inference(image: np.ndarray):
    """
    Helper function to run the blocking YOLO inference.
    We will run this in a separate thread to avoid blocking the main server loop.
    """
    results = yolo_model(image)
    detected = set()
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = yolo_model.names[class_id]
            if class_name in KNOWN_FOOD_CLASSES:
                detected.add(class_name)
    return list(detected)

# endpoints
@app.post("/detect")
async def detect_ingredients(file: UploadFile = File(...)):
    """
    Phase 1: Detect ingredients (Optimized for Concurrency)
    """
    # 1. validate the file type
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG, PNG, WEBP allowed.")

    # 2. validate size - memory crash is possible
    # check size before reading all bytes
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Max 10MB.")

    try:
        # 3. decode image
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
             raise HTTPException(status_code=400, detail="Could not decode image.")

        # 4. run inference in ThreadPool (Non-Blocking!)
        # keeps server responsive to other users while processing current image.
        detected_ingredients = await run_in_threadpool(run_yolo_inference, img)

        return {
            "message": "Detection successful",
            "detected_count": len(detected_ingredients),
            "ingredients": detected_ingredients
        }

    except Exception as e:
        print(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error during detection")

@app.post("/generate-recipe", response_model=RecipeResponse)
async def generate_recipe(request: RecipeRequest):
    """
    Phase 2: Generate Recipe using Google Gemini
    """
    if not GEMINI_KEY:
        raise HTTPException(status_code=500, detail="Server misconfigured: Missing Gemini API Key")

    # logic
    # If the user says "Any" or "Random", AI will decide.
    # Otherwise, we force the choice.
    cuisine_instruction = ""
    if request.cuisine and request.cuisine.lower() not in ["any", "random"]:
        cuisine_instruction = f"- Cuisine Style: Strictly {request.cuisine}"
    else:
        cuisine_instruction = "- Cuisine Style: Choose the best culinary style that fits the ingredients."

    # prompt
    prompt = f"""
    You are SnapRecipe, an expert AI chef.
    Create a unique recipe based on these parameters:
    
    - Available Ingredients: {', '.join(request.ingredients)}
    - Skill Level: {request.difficulty}
    - Dietary Restrictions: {request.dietary_restrictions if request.dietary_restrictions else "None"}
    {cuisine_instruction} 
    
    Requirements:
    1. Output MUST be valid JSON matching the schema.
    2. Use as many available ingredients as possible.
    3. Assume basic pantry staples (oil, salt, pepper, water) are available.
    4. If crucial ingredients are missing, list them in 'missing_ingredients'.
    5. Keep steps clear and actionable.
    """

    try:
        response = await gemini_model.generate_content_async(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=RecipeResponse
            )
        )
        
        return json.loads(response.text)

    except Exception as e:
        print(f"Gemini Error: {e}")
        raise HTTPException(status_code=503, detail="AI Chef is currently busy. Please try again.")

@app.get("/")
def root():
    return {"message": "SnapRecipe Backend is active", "docs": "/docs"}
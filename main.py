from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import pytesseract
import io

from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe" # For Windows

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_origins=["https://pranav3782.github.io/SPA_front/"],
)

llm = ChatGroq(
    base_url="https://api.groq.com", # <--- CHANGE THIS LINE
    api_key="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",
    model="llama-3.3-70b-versatile"
)

class AnalyzeRequest(BaseModel):
    ingredients: str
    product_type: str

@app.post("/extract")
async def extract_ingredients(image: UploadFile = File(...), product_type: str = Form(...)):
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents))
        text = pytesseract.image_to_string(img)
        print("OCR Extracted Text:", text)
        if not text.strip():
            return {"ingredients": "", "warning": "No text extracted. Check image quality."}
        return {"ingredients": text.strip()}
    except Exception as e:
        return {"error": str(e)}

# main.py

# ... (imports and FastAPI app setup remain the same)

@app.post("/analyze")
async def analyze_ingredients(request: AnalyzeRequest):
    prompt = f"""
You are an expert cosmetic chemist and skincare/haircare specialist. Analyze these ingredients for a {request.product_type} product:

Ingredients: {request.ingredients}

Provide your analysis in the following clear bullet-point format:

 1. Harmful Ingredients ❌:
    [Ingredient Name]: [Concise reason why it's harmful, e.g., "can cause irritation," "known allergen."]

* 2. Beneficial Ingredients ✅:
    * [Ingredient Name]: [Concise reason why it's beneficial, e.g., "hydrates skin," "provides antioxidant protection," "cleanses effectively."]*

* 3. Neutral/Conditional Ingredients ⚠️:
    * [Ingredient Name]: [Concise reason for classification, e.g., "emulsifier, generally safe," "thickener, no direct benefit/harm."]*

* 4. Suitability Recommendation 🎯:
    * Recommended for: [Skin/hair types or conditions that would benefit, e.g., "oily skin," "dry, damaged hair," "all skin types."]
    * Avoid if: [Skin/hair types or conditions that should avoid, e.g., "sensitive skin," "acne-prone skin," "color-treated hair."]
    * General Tips: [Any important usage notes or additional advice.]

Ensure your language is clear, professional, and easy to understand for a general audience.

"""
    try:
        print("Prompt sent to LLM:", prompt)
        response = llm([HumanMessage(content=prompt)])
        print("LLM response:", response.content)
        return {"result": response.content}
    except Exception as e:
        return {"result": f"Analysis failed: {e}"}

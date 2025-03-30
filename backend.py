from fastapi import FastAPI
import joblib
import openai
import os
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# ✅ Initialize FastAPI
app = FastAPI()

# ✅ Fix CORS to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change this in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# ✅ Load trained query classification model
model = joblib.load("query_classification_model.pkl")

# ✅ Load dataset to recreate TF-IDF vectorizer
df = pd.read_csv("tacticone_dataset.csv")
vectorizer = TfidfVectorizer()
vectorizer.fit(df["processed_query"])  # Fit on full dataset

# ✅ Set up OpenAI API client
openai.api_key = openai_api_key  # Fix: Correct API Key Usage

# ✅ AI-powered response function
def generate_ai_response(query):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful support assistant."},
                {"role": "user", "content": query}
            ]
        )
        return response.choices[0].message.content  # Fix: Correct way to access response
    except Exception as e:
        return f"Error: {str(e)}"

# ✅ Query classification function
def classify_and_respond(query):
    processed_query = query.lower()
    query_tfidf = vectorizer.transform([processed_query])
    predicted_category = model.predict(query_tfidf)[0]

    # ✅ Always use AI-generated response instead of rule-based ones
    ai_response = generate_ai_response(query)
    
    return {"category": predicted_category, "response": ai_response}

# ✅ Define API request model
class QueryRequest(BaseModel):
    query: str

# ✅ Define API endpoint for chat
@app.post("/chat/")
async def chat(request: QueryRequest):
    response = generate_ai_response(request.query)
    
    # ✅ Preserve newlines and add an email link at the bottom
    formatted_response = response.replace("\n", "<br>")  
    formatted_response += '<br><br><a href="mailto:support@goosebump.ai" style="color: blue; text-decoration: underline;">Contact Advanced Support</a>'

    return {"response": formatted_response}


# ✅ Run the FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

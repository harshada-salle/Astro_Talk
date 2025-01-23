from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from huggingface_hub import InferenceClient
import logging

# Set up FastAPI application and enable CORS
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Logging setup
logging.basicConfig(level=logging.DEBUG)

# Your Hugging Face API Key and model details
API_KEY = "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"
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
client = InferenceClient(token=API_KEY)

# Define input schema for POST requests
class BirthdateRequest(BaseModel):
    birthdate: str = Field(..., example="1990-01-01", description="Format: YYYY-MM-DD")
    zodiac_sign: str = Field(..., description="The user's zodiac sign")

# Root Route (Optional)
@app.get("/")
def read_root():
    return {"message": "Welcome to the Horoscope API! Please use /daily-horoscope for predictions."}

# Define the GET route to handle prediction requests with query parameters
@app.get("/daily-horoscope/")
def get_daily_horoscope_get(
    birthdate: str = Query(..., description="The user's birthdate in YYYY-MM-DD format"),
    zodiac_sign: str = Query(..., description="The user's zodiac sign")
):
    logging.debug(f"GET request received: birthdate={birthdate}, zodiac_sign={zodiac_sign}")
    return generate_horoscope(birthdate, zodiac_sign)

# Define the POST route to handle prediction requests
@app.post("/daily-horoscope/")
def get_daily_horoscope_post(request: BirthdateRequest):
    logging.debug(f"POST request received: {request}")
    return generate_horoscope(request.birthdate, request.zodiac_sign)

# Function to generate horoscope using the Hugging Face Inference API
def generate_horoscope(birthdate: str, zodiac_sign: str):
    try:
        # System message and prompt for the model
        system_message = (
            "You are an expert astrologer, providing personalized horoscopes based on the user's birthdate and zodiac sign. "
            "Be insightful, clear, and include advice on personal growth, relationships, and career."
        )

        # Make sure the prompt avoids including date ranges for zodiac signs
        prompt = (
            f"{system_message}\n\n"
            f"User: I was born on {birthdate}. My zodiac sign is {zodiac_sign}. What is my horoscope for today?\n\n"
            "Astrologer: Provide a concise horoscope for today in 3 lines."
        )

        # Generate prediction using Hugging Face Inference API
        response = client.text_generation(prompt, model=MODEL_NAME, max_new_tokens=200)

        # Debug: print the response to see if it's a string
        print(response)

        # Handle the response as a string
        if isinstance(response, str):
            horoscope = response.strip()
        else:
            horoscope = "No horoscope generated"

        # Remove any unwanted newline characters (including escape sequences like \n) and replace them with a space
        horoscope = horoscope.replace("\n", " ").strip()

        # Shorten the horoscope to 3 lines (if the generated response is too long)
        horoscope_lines = horoscope.split(".")  # We can split at periods for a clean short result
        shortened_horoscope = ". ".join(horoscope_lines[:3])  # Limit to the first 3 sentences

        # Return the response with the shortened horoscope
        return {
            "birthdate": birthdate,
            "zodiac_sign": zodiac_sign,
            "horoscope": shortened_horoscope
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI app when executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

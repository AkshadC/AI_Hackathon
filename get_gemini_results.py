import os

import requests


def get_response():
    response = requests.post(
        "https://hackathon-api-39535212257.northamerica-northeast2.run.app/api/generate",
        headers={"X-API-Key": os.getenv("GEMINI_API_KEY")},
        json={"contents": "What is the capital of France?"}
    )
    print(response.text)

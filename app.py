from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class Query(BaseModel):
    message: str

def your_python_function(msg: str):
    return f"You searched for: {msg}"

@app.post("/analyze")
def analyze(query: Query):
    result = your_python_function(query.message)
    return {"reply": result}

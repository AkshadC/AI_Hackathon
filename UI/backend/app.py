from datetime import datetime

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import time

app = FastAPI()

class SummarizeIn(BaseModel):
    query: str

class SummarizeOut(BaseModel):
    summary: str
    bullets: List[str]

class RelatedIn(BaseModel):
    text: str
    k: int = 10

class Post(BaseModel):
    title: str
    url: Optional[str] = None
    subreddit: str = "r/canada"
    score: Optional[int] = None
    num_comments: Optional[int] = None
    created_utc: Optional[str] = None
    snippet: Optional[str] = None
    similarity: Optional[float] = None

class RelatedOut(BaseModel):
    posts: List[Post]

@app.post("/summarize", response_model=SummarizeOut)
def summarize(inp: SummarizeIn):
    # TODO: call your Gemini / model inference here
    # example output:
    bullets = [
        "Point 1 about the topic...",
        "Point 2 about the topic...",
        "Point 3 about the topic...",
    ]
    summary = " ".join(bullets)
    return SummarizeOut(summary=summary, bullets=bullets)

@app.post("/related_posts", response_model=RelatedOut)
def related(req: RelatedIn):
    today = datetime.now().strftime("%Y-%m-%d")

    # ✅ show the input in title/snippet to verify it's coming through
    return RelatedOut(posts=[
        Post(
            title=f"[Discussion] Related to: {req.text[:80]}",
            subreddit="r/canada",
            score=1200,
            num_comments=450,
            created=today,
            snippet=f"Backend received text: {req.text[:120]}",
            similarity=0.82,
            url="https://reddit.com"
        )
    ])

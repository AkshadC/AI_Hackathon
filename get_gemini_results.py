import json
import os

import requests
key = os.getenv("GEMINI_API_KEY")
API_URL = "https://hackathon-api-39535212257.northamerica-northeast2.run.app/api/generate"

def call_gemini(prompt: str) -> dict:
    r =requests.post(
    "https://hackathon-api-39535212257.northamerica-northeast2.run.app/api/generate",
    headers={"X-API-Key": os.getenv("GEMINI_API_KEY")},
    json={"contents": prompt}
)
    # API may return dict or a string; handle both
    data = r.json()
    if isinstance(data, dict):
        # common cases: {"text": "..."} or {"candidates":[...]} etc.
        return data
    return {"text": str(data)}

def extract_text(resp_json: dict) -> str:
    # Try a few common shapes
    if "text" in resp_json and isinstance(resp_json["text"], str):
        return resp_json["text"]
    if "output" in resp_json and isinstance(resp_json["output"], str):
        return resp_json["output"]
    if "candidates" in resp_json:
        try:
            return resp_json["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            pass
    return json.dumps(resp_json)  # fallback

def build_community_prompt(query: str, comment_clusters: list) -> str:
    # comment_clusters from your print loop:
    # [{"cluster_id":..., "size":..., "rep_comments":[{"comment_id":..., "text":...}, ...]}, ...]
    payload = {"query": query, "comment_clusters": comment_clusters}
    return f"""
Summarize what Reddit users are saying about the query using ONLY the provided clustered comments.
Return plain text (no JSON), 5-10 bullets max.
- Identify 3-6 themes/viewpoints
- Mention overall tone (positive/negative/mixed/neutral)
- Cite evidence by including comment_id in parentheses for key bullets.

INPUT:
{json.dumps(payload, ensure_ascii=False)}
""".strip()

def build_urls_prompt(query: str, top_threads: list, url_summaries: list) -> str:
    # url_summaries: [{"thread_id":..., "url":..., "text":...}, ...]
    payload = {"query": query, "threads": top_threads, "articles": url_summaries}
    return f"""
Summarize the factual information from external URLs about the query using ONLY the provided article texts.
Return plain text (no JSON), 5-10 bullets max.
- If some article text is empty, ignore it
- Do NOT mention Reddit comments

INPUT:
{json.dumps(payload, ensure_ascii=False)}
""".strip()

def fetch_url_text(url: str, timeout=10, max_chars=8000) -> str:
    # super light fetch (no heavy parsing) – replace with your cleaner if you have one
    import requests
    from bs4 import BeautifulSoup

    if not isinstance(url, str) or not url.strip():
        return ""
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200 or not r.text:
            return ""
        soup = BeautifulSoup(r.text, "lxml")
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
            tag.decompose()
        text = " ".join(soup.get_text(" ", strip=True).split())
        return text[:max_chars]
    except Exception:
        return ""


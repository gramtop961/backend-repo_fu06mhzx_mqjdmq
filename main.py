import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import math
import requests

from database import db, create_document, get_documents
from schemas import TeamRating, PredictionRequest, PredictionResult

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Soccer Predictor API"}

@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
    return response

# --- Simple odds model ---
# We use an Elo-style pre-match rating and adjust by current score + minute + home advantage.
# This is intentionally lightweight; you can later plug in richer features.

def logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

HOME_ADVANTAGE = 60.0  # Elo points
GOAL_VALUE = 50.0      # Elo-equivalent per net goal lead
MINUTE_DECAY = 0.025   # Confidence scaling as game progresses

# Store and update team ratings
@app.post("/api/teams/rating")
def upsert_team_rating(team: TeamRating):
    if db is None:
        raise HTTPException(500, "Database not configured")
    existing = db["teamrating"].find_one({"team_name": team.team_name})
    doc = team.model_dump()
    if existing:
        db["teamrating"].update_one({"_id": existing["_id"]}, {"$set": doc})
        return {"status": "updated"}
    else:
        create_document("teamrating", doc)
        return {"status": "created"}

@app.get("/api/teams/rating")
def get_team_rating(team_name: str):
    if db is None:
        raise HTTPException(500, "Database not configured")
    existing = db["teamrating"].find_one({"team_name": team_name})
    if not existing:
        return {"team_name": team_name, "rating": 1500.0}
    return {"team_name": existing["team_name"], "rating": float(existing.get("rating", 1500.0))}

@app.post("/api/predict", response_model=PredictionResult)
def predict(req: PredictionRequest):
    # Fetch ratings
    if db is not None:
        hdoc = db["teamrating"].find_one({"team_name": req.home_team})
        adoc = db["teamrating"].find_one({"team_name": req.away_team})
        r_home = float(hdoc.get("rating", 1500.0)) if hdoc else 1500.0
        r_away = float(adoc.get("rating", 1500.0)) if adoc else 1500.0
    else:
        r_home, r_away = 1500.0, 1500.0

    # Base difference
    diff = r_home - r_away
    # Home advantage
    if not req.is_neutral:
        diff += HOME_ADVANTAGE
    # Current score impact
    diff += (req.home_score - req.away_score) * GOAL_VALUE
    # Minute weighting: later minutes increase certainty
    minute_factor = logistic((req.minute - 45) * MINUTE_DECAY)
    effective_diff = diff * (0.7 + 0.6 * minute_factor)  # scales from ~0.7x early to ~1.3x late

    # Convert diff to probabilities using logistic; add draw probability via softness window
    p_home_raw = logistic(effective_diff / 400.0 * math.log(10))
    p_away_raw = 1.0 - p_home_raw

    # Draw modeling: higher draw chance when effective diff is small and as minute increases
    closeness = math.exp(-abs(effective_diff) / 200.0)
    late_draw_boost = 0.1 + 0.2 * logistic((req.minute - 60) * 0.05)
    p_draw = min(0.55, closeness * late_draw_boost)

    # Re-normalize
    scale = 1.0 - p_draw
    p_home = p_home_raw * scale
    p_away = p_away_raw * scale

    # Return normalized
    total = p_home + p_draw + p_away
    if total <= 0:
        raise HTTPException(400, "Invalid probabilities computed")
    p_home /= total
    p_draw /= total
    p_away /= total

    return PredictionResult(
        p_home=round(p_home, 4),
        p_draw=round(p_draw, 4),
        p_away=round(p_away, 4),
        effective_diff=round(effective_diff, 2)
    )

# Optional: basic scraping hook for rezultati.com (SofaScore/Flashscore style)
# We will not scrape directly to avoid ToS issues; instead we accept a match URL and
# return a friendly message. In a real deployment, you'd integrate via allowed APIs.
class MatchLink(BaseModel):
    url: str

@app.post("/api/match-link")
def ingest_match_link(link: MatchLink):
    # Just store link for reference
    try:
        create_document("matchlink", {"url": link.url})
    except Exception:
        pass
    return {"status": "received", "url": link.url}

# Placeholder for video upload; we won't run full CV here but allow client to upload and
# associate with a match. You can later plug in a ML endpoint.
@app.post("/api/upload-video")
def upload_video(file: UploadFile = File(...), match_id: Optional[str] = Form(None)):
    filename = file.filename
    size = 0
    while True:
        chunk = file.file.read(1024 * 1024)
        if not chunk:
            break
        size += len(chunk)
    return {"status": "stored-temp", "filename": filename, "size": size, "match_id": match_id}

# Public schema exposure for tooling
@app.get("/schema")
def get_schema():
    return {
        "TeamRating": TeamRating.model_json_schema(),
        "PredictionRequest": PredictionRequest.model_json_schema(),
        "PredictionResult": PredictionResult.model_json_schema(),
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

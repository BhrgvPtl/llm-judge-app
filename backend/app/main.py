from typing import Literal, List, Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .engine import generate_answers_for_task, peer_review_all
from .aggregator import aggregate_candidates
from .config_loader import get_available_tasks
from .schemas import task_list


app = FastAPI(
    title="LLM Judge App",
    description="Runs multiple HF models and aggregates a final answer.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SolveRequest(BaseModel):
    task: Literal[
        "math",
        "code",
        "research",
        "qa",
        "creative",
        "summary",
        "business",
        "data",
        "translation",
        "chat",
    ]
    prompt: str


class CandidateResponse(BaseModel):
    model_id: str
    candidate_id: int
    text: str
    peer_scores: Dict[str, float]
    peer_explanations: Dict[str, str]


class SolveResponse(BaseModel):
    final_answer: str
    meta: Dict[str, Any]
    candidates: List[CandidateResponse]


@app.get("/tasks")
def list_tasks():
    available = set(get_available_tasks())
    all_tasks = [t for t in task_list() if t["id"] in available]
    return {"tasks": all_tasks}


@app.post("/solve", response_model=SolveResponse)
def solve(req: SolveRequest):
    # 1) generate 1 answer per model (2 models total)
    candidates = generate_answers_for_task(req.task, req.prompt)

    # 2) (currently a no-op, but keeps the API shape)
    candidates = peer_review_all(req.task, req.prompt, candidates)

    # 3) aggregate into one final answer using the small meta model
    aggregated = aggregate_candidates(req.task, req.prompt, candidates)

    candidates_payload = [
        CandidateResponse(
            model_id=c.model_id,
            candidate_id=c.candidate_id,
            text=c.text,
            peer_scores=c.peer_scores,
            peer_explanations=c.peer_explanations,
        )
        for c in candidates
    ]

    return SolveResponse(
        final_answer=aggregated.final_answer,
        meta={"note": "Lightweight meta aggregation using a small meta model."},
        candidates=candidates_payload,
    )

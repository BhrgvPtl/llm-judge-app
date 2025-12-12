from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class ModelConfig:
    id: str
    task: str
    max_tokens: int = 512
    temperature: float = 0.7


@dataclass
class CandidateAnswer:
    model_id: str
    candidate_id: int
    text: str
    peer_scores: Dict[str, float]  # judge_model_id -> score
    peer_explanations: Dict[str, str]  # judge_model_id -> reason string


@dataclass
class AggregatedResult:
    final_answer: str
    chosen_fragments: Dict[str, Any]


# Just for convenience; your UI will use these
TASK_LABELS: Dict[str, str] = {
    "math": "Math problem solving",
    "code": "Coding / code generation",
    "research": "Research / literature review",
    "qa": "General Q&A / tutoring",
    "creative": "Creative writing",
    "summary": "Summarization / note taking",
    "business": "Business / professional writing",
    "data": "Data analysis / stats explanation",
    "translation": "Translation / multilingual",
    "chat": "Open chat / brainstorming",
}


def task_list() -> List[Dict[str, str]]:
    return [{"id": k, "label": v} for k, v in TASK_LABELS.items()]

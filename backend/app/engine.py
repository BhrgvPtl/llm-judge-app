import re
from typing import List
from .models import get_text_generation_pipeline
from .config_loader import get_models_for_task
from .schemas import CandidateAnswer, ModelConfig

TOP_K_MODELS = 5 
NUM_CANDIDATES_PER_MODEL = 1 

def _strip_prompt(full_text: str, prompt: str) -> str:
    if full_text.startswith(prompt):
        return full_text[len(prompt):].strip()
    return full_text.strip()

def generate_answers_for_task(task: str, user_prompt: str) -> List[CandidateAnswer]:
    # Select best 5 small models
    model_configs: List[ModelConfig] = get_models_for_task(task)[:TOP_K_MODELS]
    all_candidates: List[CandidateAnswer] = []

    print(f"Generating drafts with {len(model_configs)} small models...")

    for mc in model_configs:
        try:
            pipe = get_text_generation_pipeline(mc.id)
            
            # Simple prompt for small models
            sys_prompt = (
                f"Task: {task}\n"
                f"Question: {user_prompt}\n"
                "Answer concisely and accurately:"
            )

            outputs = pipe(
                sys_prompt,
                max_new_tokens=mc.max_tokens,
                num_return_sequences=NUM_CANDIDATES_PER_MODEL,
                do_sample=True,
                temperature=mc.temperature
            )

            for idx, out in enumerate(outputs):
                text = _strip_prompt(out['generated_text'], sys_prompt)
                all_candidates.append(
                    CandidateAnswer(
                        model_id=mc.id,
                        candidate_id=idx,
                        text=text,
                        peer_scores={},
                        peer_explanations={},
                    )
                )
        except Exception as e:
            print(f"Skipping {mc.id}: {e}")
            continue

    return all_candidates

def peer_review_all(task: str, user_prompt: str, candidates: List[CandidateAnswer]) -> List[CandidateAnswer]:
    """
    Small models review each other.
    """
    if not candidates:
        return candidates

    participating_model_ids = list(set(c.model_id for c in candidates))
    print("Running Peer Review...")

    for judge_model_id in participating_model_ids:
        try:
            pipe = get_text_generation_pipeline(judge_model_id)
            
            for cand in candidates:
                if cand.model_id == judge_model_id:
                    continue
                
                # Simplified scoring prompt for small models
                scoring_prompt = (
                    f"Evaluate this answer for {task}.\n"
                    f"Question: {user_prompt}\n"
                    f"Answer: {cand.text[:500]}...\n" # Truncate for speed
                    "Score (0-10) and Reason.\n"
                    "Format: Score: 5\nReason: Good but short."
                )

                outputs = pipe(
                    scoring_prompt,
                    max_new_tokens=40, # Short output for speed
                    num_return_sequences=1,
                    temperature=0.1
                )

                full = outputs[0]["generated_text"]
                resp = _strip_prompt(full, scoring_prompt)

                score_match = re.search(r"Score:\s*([0-9]+(?:\.[0-9]+)?)", resp, re.IGNORECASE)
                reason_match = re.search(r"Reason:\s*(.+)", resp, re.IGNORECASE | re.DOTALL)

                score = float(score_match.group(1)) if score_match else 5.0
                reason = reason_match.group(1).strip() if reason_match else "No reason provided."

                cand.peer_scores[judge_model_id] = score
                cand.peer_explanations[judge_model_id] = reason
                
        except Exception as e:
            print(f"Peer review failed for {judge_model_id}: {e}")
            continue

    return candidates
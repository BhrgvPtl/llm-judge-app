from typing import List
from .models import get_text_generation_pipeline
from .config_loader import get_meta_model_config
from .schemas import CandidateAnswer, AggregatedResult

def _strip_prompt(full_text: str, prompt: str) -> str:
    if full_text.startswith(prompt):
        return full_text[len(prompt):].strip()
    return full_text.strip()

def aggregate_candidates(task: str, user_prompt: str, candidates: List[CandidateAnswer]) -> AggregatedResult:
    """
    Uses the "Big" Meta Model to synthesize the final answer.
    """
    if not candidates:
        return AggregatedResult(final_answer="No answers generated.", chosen_fragments={})

    # Load the Big Model (e.g., Llama-3.1-8B)
    meta_cfg = get_meta_model_config()
    pipe = get_text_generation_pipeline(meta_cfg.id)

    # Build prompt with Answer + Peer Reviews
    parts = []
    for idx, c in enumerate(candidates, start=1):
        # Calculate average peer score
        scores = c.peer_scores.values()
        avg_score = sum(scores) / len(scores) if scores else 0
        
        parts.append(f"--- Option {idx} (Score: {avg_score:.1f}/10) ---")
        parts.append(f"Content:\n{c.text.strip()}\n")
        parts.append("Peer Feedback:")
        for judge, reason in c.peer_explanations.items():
            # Truncate feedback to save tokens
            parts.append(f"  - {judge}: {reason[:80]}...")
        parts.append("\n")

    answers_block = "\n".join(parts)

    prompt = (
        f"You are the Chief Editor. Your goal is to write the single best response to the user.\n"
        f"User Task: {task}\n"
        f"User Question: {user_prompt}\n\n"
        "Here are drafts from your junior team (Options 1-5), along with their peer review scores:\n"
        f"{answers_block}\n"
        "Instructions:\n"
        "1. Ignore low-quality options (low peer scores).\n"
        "2. Combine the best insights from the high-scoring options.\n"
        "3. Write a coherent, professional final answer. Do not just list the options.\n"
        "4. Do NOT mention 'Option 1' or 'Model X' in your final text.\n\n"
        "Final Answer:"
    )

    outputs = pipe(
        prompt,
        max_new_tokens=meta_cfg.max_tokens,
        temperature=0.3,
        num_return_sequences=1,
    )

    full_text = outputs[0]["generated_text"]
    final = _strip_prompt(full_text, prompt)

    return AggregatedResult(
        final_answer=final.strip(),
        chosen_fragments={},
    )
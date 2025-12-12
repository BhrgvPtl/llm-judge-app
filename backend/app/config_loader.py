from pathlib import Path
from typing import List
import yaml

from .schemas import ModelConfig

_CONFIG_CACHE = None

CONFIG_PATH = Path(__file__).parent / "models.yaml"


def _load_config() -> dict:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            _CONFIG_CACHE = yaml.safe_load(f)
    return _CONFIG_CACHE


def get_models_for_task(task: str) -> List[ModelConfig]:
    cfg = _load_config()
    tasks_cfg = cfg.get("tasks", {})
    task_cfg = tasks_cfg.get(task)
    if not task_cfg:
        raise ValueError(f"Unknown task: {task}")

    default_max_tokens = int(task_cfg.get("default_max_tokens", 512))
    default_temperature = float(task_cfg.get("default_temperature", 0.7))

    result: List[ModelConfig] = []
    for m in task_cfg.get("models", []):
        model_id = m["id"]
        max_tokens = int(m.get("max_tokens", default_max_tokens))
        temperature = float(m.get("temperature", default_temperature))
        result.append(
            ModelConfig(
                id=model_id,
                task=task,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        )
    return result


def get_available_tasks() -> List[str]:
    cfg = _load_config()
    return list(cfg.get("tasks", {}).keys())


def get_meta_model_config() -> ModelConfig:
    cfg = _load_config()
    meta_cfg = cfg.get("meta_model")
    if not meta_cfg:
        raise ValueError("meta_model section missing in models.yaml")

    return ModelConfig(
        id=meta_cfg["id"],
        task="meta",
        max_tokens=int(meta_cfg.get("max_tokens", 1024)),
        temperature=float(meta_cfg.get("temperature", 0.3)),
    )

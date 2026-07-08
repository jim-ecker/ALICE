"""NLI entailment signal via a local MNLI cross-encoder.

Premise = the sentence covering the evidence span, hypothesis = the verbalized
triple. Unlike the lexical alignment checks, an NLI model detects negation,
wrong relation direction, and misattribution. Requires the `trust` extra
(`uv sync --extra trust`); degrades to None when unavailable.
"""

from __future__ import annotations

import importlib.util
import logging
import threading

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_model = None
_tokenizer = None
_loaded_id: str | None = None
_unavailable = False


def nli_available() -> bool:
    return (
        importlib.util.find_spec("transformers") is not None
        and importlib.util.find_spec("torch") is not None
    )


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load(model_id: str, device: str) -> bool:
    global _model, _tokenizer, _loaded_id, _unavailable
    if _unavailable:
        return False
    if _model is not None and _loaded_id == model_id:
        return True
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(model_id)
        _model = AutoModelForSequenceClassification.from_pretrained(model_id)
        _model.to(_resolve_device(device))
        _model.eval()
        _loaded_id = model_id
        return True
    except Exception:
        logger.exception("NLI model %s failed to load; nli signals disabled for this run", model_id)
        _unavailable = True
        _model = None
        _tokenizer = None
        return False


def score_entailment(
    premise: str,
    hypothesis: str,
    *,
    model_id: str,
    device: str = "auto",
    max_premise_chars: int = 1000,
) -> tuple[float, float] | None:
    """(p_entailment, p_contradiction) softmax probabilities, or None if the
    trust extra isn't installed or the model failed to load."""
    if not premise or not hypothesis or not nli_available():
        return None

    # Single lock for load and inference: the model is a process-wide singleton
    # shared by the extraction worker threads.
    with _lock:
        if not _load(model_id, device):
            return None
        try:
            import torch

            inputs = _tokenizer(
                premise[:max_premise_chars],
                hypothesis,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(_model.device)
            with torch.no_grad():
                probs = torch.softmax(_model(**inputs).logits[0], dim=-1)

            # Label order differs between MNLI checkpoints — resolve by name
            p_entail = 0.0
            p_contra = 0.0
            for idx, label in _model.config.id2label.items():
                name = label.lower()
                if name.startswith("entail"):
                    p_entail = float(probs[int(idx)])
                elif name.startswith("contradict"):
                    p_contra = float(probs[int(idx)])
            return p_entail, p_contra
        except Exception:
            logger.exception("NLI inference failed; nli signals disabled for this run")
            global _unavailable
            _unavailable = True
            return None

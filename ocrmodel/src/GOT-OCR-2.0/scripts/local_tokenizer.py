"""Offline tokenizer loading helpers shared by validation entry points.

The GOT/Qwen tokenizer may be represented by ``qwen.tiktoken`` and custom
tokenizer code rather than one of the standard ``tokenizer.json`` files.  The
Transformers loader is the source of truth for deciding whether a local
directory is usable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


def tokenizer_candidates(
    primary: Path,
    fallback: str | Path | None = None,
) -> tuple[Path, ...]:
    """Return ordered, deduplicated local tokenizer directories."""

    candidates: list[Path] = []
    seen: set[Path] = set()
    for value in (primary, fallback):
        if value is None or not str(value).strip():
            continue
        candidate = Path(value).expanduser().resolve()
        if candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)
    return tuple(candidates)


def _exception_summary(error: Exception, limit: int = 240) -> str:
    message = " ".join(str(error).split())
    if len(message) > limit:
        message = message[: limit - 3] + "..."
    detail = type(error).__name__
    return f"{detail}: {message}" if message else detail


def load_local_tokenizer(
    auto_tokenizer: Any,
    candidates: Iterable[Path],
    **kwargs: Any,
) -> tuple[Any, Path]:
    """Load the first usable local tokenizer using Transformers itself.

    ``AutoTokenizer`` knows about both standard and custom GOT/Qwen tokenizer
    layouts.  Failed candidates are retained in a bounded error so a missing
    or incompatible fallback is diagnosable without dumping a full traceback.
    """

    attempted: list[str] = []
    for raw_candidate in candidates:
        candidate = Path(raw_candidate).expanduser().resolve()
        if not candidate.is_dir():
            attempted.append(f"{candidate} (directory does not exist)")
            continue
        try:
            tokenizer = auto_tokenizer.from_pretrained(str(candidate), **kwargs)
        except Exception as error:  # Transformers raises several loader-specific types.
            attempted.append(f"{candidate} ({_exception_summary(error)})")
            continue
        return tokenizer, candidate

    if attempted:
        detail = "; ".join(attempted)
    else:
        detail = "no tokenizer candidates were supplied"
    raise RuntimeError(
        "Unable to load a local tokenizer with AutoTokenizer.from_pretrained. "
        f"Tried: {detail}"
    )

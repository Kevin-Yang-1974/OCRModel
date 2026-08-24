"""Prediction schema and deterministic output normalization."""

from dataclasses import asdict, dataclass
from typing import Any
import json
import re


@dataclass
class PredictionRecord:
    page_id: str
    image: str
    model: str
    raw_output: Any
    normalized_text: str
    status: str
    runtime: dict[str, Any]
    layout: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _json_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("text", "markdown", "content", "rec_text", "overall_ocr_res"):
            if key in value:
                candidate = _json_text(value[key])
                if candidate is not None:
                    return candidate
        if isinstance(value.get("res"), dict):
            return _json_text(value["res"])
        for list_key in ("recognition_results", "results"):
            if isinstance(value.get(list_key), list):
                return "\n".join(filter(None, (_json_text(item) for item in value[list_key])))
        if isinstance(value.get("rec_texts"), list):
            return "\n".join(str(item) for item in value["rec_texts"])
    if isinstance(value, list):
        return "\n".join(filter(None, (_json_text(item) for item in value)))
    return None


def normalize_text(raw_output: Any) -> str:
    """Extract text without looking at references or applying model-specific tuning."""
    text = _json_text(raw_output)
    if text is None:
        text = str(raw_output) if raw_output is not None else ""
    text = re.sub(r"^```(?:markdown|md|text|html)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"<(?:p|div|span|li|h[1-6])[^>]*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</(?:p|div|span|li|h[1-6])>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\r\n?", "\n", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def jsonl_record(record: PredictionRecord) -> str:
    return json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True)

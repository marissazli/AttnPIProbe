import json
from pathlib import Path


def _load_json_or_jsonl(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")

    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as f_in:
            for line_no, line in enumerate(f_in, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSONL at {path}:{line_no}: {exc}"
                    ) from exc
        return records

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f_in:
            return json.load(f_in)

    raise ValueError(
        f"Unsupported file extension for {path}. Use .json or .jsonl."
    )


def load_local_queries(path):
    data = _load_json_or_jsonl(path)
    if not isinstance(data, list):
        raise ValueError("Local queries file must contain a JSON array or JSONL records.")

    required_fields = {"id", "text", "label"}
    normalized = []
    for idx, record in enumerate(data):
        if not isinstance(record, dict):
            raise ValueError(f"Query record at index {idx} must be an object.")
        missing = required_fields - set(record.keys())
        if missing:
            raise ValueError(
                f"Query record at index {idx} is missing fields: {sorted(missing)}"
            )
        normalized.append(
            {
                "id": str(record["id"]),
                "text": str(record["text"]),
                "label": int(record["label"]),
                "adversarial_text": str(record.get("adversarial_text", "")),
                "source_prompt_id": str(record.get("source_prompt_id", "")),
                "variant_index": int(record.get("variant_index", 0))
                if str(record.get("variant_index", "")).strip()
                else 0,
                "attack_type": str(record.get("attack_type", "")),
                "motivation": str(record.get("motivation", "")),
            }
        )
    return normalized


def load_local_corpus(path):
    data = _load_json_or_jsonl(path)
    if not isinstance(data, list):
        raise ValueError("Local corpus file must contain a JSON array or JSONL records.")

    required_fields = {"doc_id", "text"}
    normalized = []
    for idx, record in enumerate(data):
        if not isinstance(record, dict):
            raise ValueError(f"Corpus record at index {idx} must be an object.")
        missing = required_fields - set(record.keys())
        if missing:
            raise ValueError(
                f"Corpus record at index {idx} is missing fields: {sorted(missing)}"
            )
        normalized.append(
            {
                "doc_id": str(record["doc_id"]),
                "text": str(record["text"]),
                "title": str(record.get("title", "")),
            }
        )
    return normalized


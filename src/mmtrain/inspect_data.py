from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .data import normalize_record


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate multimodal JSONL before training")
    parser.add_argument("--input", required=True)
    parser.add_argument("--image-root", default="data/raw")
    parser.add_argument("--text-field", default="text")
    args = parser.parse_args()

    source = Path(args.input)
    image_root = Path(args.image_root)
    counts = Counter()
    missing = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            text, images = normalize_record(record, args.text_field)
            counts["records"] += 1
            counts["characters"] += len(text)
            counts["images"] += len(images)
            if not text.strip():
                counts["empty_text"] += 1
            for name in images:
                path = Path(name)
                if not path.is_absolute():
                    path = image_root / path
                if not path.exists():
                    missing.append({"line": line_number, "image": str(path)})
    report = dict(counts)
    report["missing_image_count"] = len(missing)
    report["missing_image_examples"] = missing[:20]
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if counts["empty_text"] or missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()


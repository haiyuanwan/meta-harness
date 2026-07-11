import json
from pathlib import Path


def main() -> None:
    expected = [
        {"id": "evt-a", "seq": 3, "status": "closed", "note": "final"},
        {"id": "evt-b", "seq": 3, "status": "closed", "note": "resolved"},
        {"id": "evt-c", "seq": 1, "status": "closed", "note": "done"},
        {"id": "evt-e", "seq": 2, "status": "open", "note": "later"},
    ]
    try:
        actual = json.loads(Path("/app/unique.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        actual = None
    Path("/logs/verifier/reward.txt").write_text("1\n" if actual == expected else "0\n")


if __name__ == "__main__":
    main()

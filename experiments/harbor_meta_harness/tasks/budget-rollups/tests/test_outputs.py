import json
from pathlib import Path


def main() -> None:
    expected = {"growth": 320, "legal": 75, "ops": 555, "research": 300}
    try:
        actual = json.loads(Path("/app/rollups.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        actual = None
    Path("/logs/verifier/reward.txt").write_text("1\n" if actual == expected else "0\n")


if __name__ == "__main__":
    main()

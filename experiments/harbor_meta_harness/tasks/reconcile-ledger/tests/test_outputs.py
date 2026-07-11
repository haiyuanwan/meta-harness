import json
from pathlib import Path


def main() -> None:
    expected = {"acme": 575, "bravo": 650, "charlie": -200}
    try:
        actual = json.loads(Path("/app/report.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        actual = None
    Path("/logs/verifier/reward.txt").write_text("1\n" if actual == expected else "0\n")


if __name__ == "__main__":
    main()

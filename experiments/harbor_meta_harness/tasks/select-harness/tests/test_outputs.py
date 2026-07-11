import json
from pathlib import Path


def main() -> None:
    try:
        report = json.loads(Path("/app/evaluation_history.json").read_text())
        history = report["history"]
        rewards = [
            entry["target"]["reward"] for entry in history if entry.get("target")
        ]
        valid = report["budget"] == 4 and report["remaining_budget"] == 4 - len(history)
        reward = max(rewards, default=0) if valid else 0
    except (FileNotFoundError, KeyError, TypeError, json.JSONDecodeError):
        reward = 0
    Path("/logs/verifier/reward.txt").write_text(f"{reward}\n")


if __name__ == "__main__":
    main()

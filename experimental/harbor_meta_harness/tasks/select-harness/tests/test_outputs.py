import json
from pathlib import Path


def main() -> None:
    try:
        report = json.loads(Path("/app/evaluation_history.json").read_text())
        history = report["history"]
        rewards = [
            float(entry["reward"])
            for entry in history
            if entry.get("accepted") and entry.get("reward") is not None
        ]
        valid = (
            report["remaining_budget"] == report["budget"] - len(history)
            and report["budget"] >= 1
        )
        reward = max(rewards, default=0.0) if valid else 0.0
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        reward = 0.0
    Path("/logs/verifier/reward.txt").write_text(f"{reward}\n")


if __name__ == "__main__":
    main()

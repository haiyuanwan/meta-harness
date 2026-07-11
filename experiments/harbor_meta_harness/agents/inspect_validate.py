from agents.terminal_harness import TerminalHarness


class AgentHarness(TerminalHarness):
    prompt = """Solve the task with the terminal.
First inspect the input and task requirements.
Then implement the requested artifact.
Run a local validation command against the artifact before you stop.
Do not claim completion until the required output file exists."""
    max_turns = 6

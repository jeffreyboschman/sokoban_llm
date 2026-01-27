from src.sokoban import SokobanState, ACTIONS


class OneStepPolicy:
    """
    Abstract interface — LLM MUST obey this contract.
    """
    def predict(self, state: SokobanState) -> list[tuple[str, float]]:
        raise NotImplementedError


class MockHeuristicPolicy(OneStepPolicy):
    """
    Replace with a real LLM later.
    """
    def predict(self, state: SokobanState) -> list[tuple[str, float]]:
        # Uniform prior (baseline)
        return [(a, 1.0) for a in ACTIONS.keys()]
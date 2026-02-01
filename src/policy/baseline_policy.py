
from src.policy.base import OneStepPolicy
from src.sokoban import SokobanState, ACTIONS
import logging

_logger = logging.getLogger(__name__)

class MockHeuristicPolicy(OneStepPolicy):
    """
    Replace with a real LLM later.
    """
    def predict(self, state: SokobanState) -> list[tuple[str, float]]:
        # Uniform prior (baseline)
        return [(a, 1.0) for a in ACTIONS.keys()]
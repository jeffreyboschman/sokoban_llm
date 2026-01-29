from src.sokoban import SokobanState, ACTIONS
import logging

_logger = logging.getLogger(__name__)

class OneStepPolicy:
    """
    Abstract interface — LLM MUST obey this contract.
    """
    def predict(self, state: SokobanState) -> list[tuple[str, float]]:
        raise NotImplementedError



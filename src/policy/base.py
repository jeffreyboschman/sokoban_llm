from src.sokoban import SokobanState, ACTIONS
import logging

_logger = logging.getLogger(__name__)

class OneStepPolicy:
    """
    Abstract interface — LLM MUST obey this contract.
    """
    def predict_and_log(self, state: SokobanState) -> list[tuple[str, float]]:
        """Predict action probabilities and log the process.

        Args:
            state (SokobanState): The current Sokoban state for which to predict action probabilities.

        Returns:
            list[tuple[str, float]]: A list of (action, probability) tuples representing the predicted action distribution.
        """
        _logger.debug("Predicting action distribution for state:\n%s", state.render())
        predictions = self.predict(state)
        _logger.debug("Raw predictions: %s", predictions)
        predicted_probs = {a: p for a, p in predictions}
        total_prob = sum(predicted_probs.values())
        normalized_predictions = [(a, p / total_prob) for a, p in predictions]
        _logger.debug("Normalized predictions: %s", normalized_predictions)
        return normalized_predictions


    def predict(self, state: SokobanState) -> list[tuple[str, float]]:
        """Abstract method to predict action probabilities.
        Args:
            state (SokobanState): The current Sokoban state for which to predict action probabilities.
        
        Returns:
            list[tuple[str, float]]: A list of (action, probability) tuples representing the predicted action distribution.
        """
        raise NotImplementedError



from __future__ import annotations
from dataclasses import dataclass, field
import math
import itertools
import networkx as nx

from src.sokoban import SokobanState
from src.policy.base import OneStepPolicy

import logging



_logger = logging.getLogger(__name__)


@dataclass
class SearchNode:
    state: SokobanState
    parent: SearchNode | None
    action: str | None
    score: float
    depth: int
    id: int = field(default_factory=itertools.count().__next__)



class BeamSearchSolver:
    """
    Beam search guided by a one-step policy (LLM).

    At each depth:
    - Expand all nodes currently in the beam
    - Collect all valid children
    - Keep only the top-K scoring children as the next beam
    """

    def __init__(
        self,
        policy: OneStepPolicy,
        beam_width: int = 4,
        max_depth: int = 50,
    ):
        self.policy = policy
        self.beam_width = beam_width
        self.max_depth = max_depth

        # Search tree for visualization / debugging
        self.graph = nx.DiGraph()

    def solve(self, start: SokobanState) -> SearchNode | None:
        """
        Perform beam search starting from `start`.

        Returns:
            SearchNode corresponding to a goal state if found,
            otherwise None.
        """

        # Root of the search tree (no action led here)
        root = SearchNode(
            state=start,
            parent=None,
            action=None,
            score=0.0,
            depth=0,
        )

        # Current beam (frontier) — starts with just the root
        frontier = [root]

        # Used to avoid revisiting identical board configurations
        visited = set()

        # Add root to visualization graph
        self.graph.add_node(root.id, label="START")

        _logger.info("Starting beam search")
        _logger.info("Beam width = %d | Max depth = %d", self.beam_width, self.max_depth)

        # Main beam search loop (depth-limited)
        for depth in range(self.max_depth):
            _logger.debug("=" * 40)
            _logger.debug("Depth %d | Frontier size: %d", depth, len(frontier))

            candidates = []  # all children from this depth

            # Expand every node currently in the beam
            for node in frontier:
                _logger.debug(
                    "Expanding node %d | depth=%d | score=%.3f",
                    node.id,
                    node.depth,
                    node.score,
                )

                # Check goal BEFORE expanding
                if node.state.is_goal():
                    _logger.info("Goal reached at depth %d", node.depth)
                    return node

                # Hashable key for visited check
                state_key = node.state.grid

                # If we've already expanded this state, skip it
                if state_key in visited:
                    _logger.debug(
                        "Skipping node %d (state already visited)", node.id
                    )
                    continue

                visited.add(state_key)

                # Query the one-step policy (LLM)
                action_scores = self.policy.predict_and_log(node.state)
                _logger.debug(
                    "Policy returned %d candidate actions", len(action_scores)
                )
                sorted_actions = sorted(
                    action_scores, key=lambda x: x[1], reverse=True
                )
                _logger.info(
                    "Top %s actions: %s",
                    self.beam_width,
                    [f"{action} ({score:.3f})" for action, score in sorted_actions[:self.beam_width]],
                )

                valid_children = 0

                # Try the top actions suggested by the policy
                for action, logit in sorted_actions[: self.beam_width]:
                    next_state = node.state.step(action)

                    # Invalid move (wall, blocked push, etc.)
                    if next_state is None:
                        _logger.debug(
                            "Action '%s' from node %d is invalid", action, node.id
                        )
                        continue

                    # Accumulate log-probability (or heuristic score)
                    score = node.score + math.log(logit + 1e-6)

                    child = SearchNode(
                        state=next_state,
                        parent=node,
                        action=action,
                        score=score,
                        depth=node.depth + 1,
                    )

                    # Add to search graph
                    self.graph.add_node(
                        child.id,
                        label=f"{action}\n{score:.2f}",
                    )
                    self.graph.add_edge(node.id, child.id)

                    candidates.append(child)
                    valid_children += 1

                # If no children were generated, this node is a dead end
                if valid_children == 0:
                    _logger.debug(
                        "Node %d is a dead end (no valid actions)", node.id
                    )
                else:
                    _logger.debug(
                        "Node %d produced %d valid children",
                        node.id,
                        valid_children,
                    )

            # If nothing to expand further, terminate early
            if not candidates:
                _logger.warning(
                    "Search terminated early at depth %d (no candidates)", depth
                )
                return None

            # Sort all candidates by score (higher = better)
            candidates.sort(key=lambda n: n.score, reverse=True)

            # Prune to beam width
            if len(candidates) > self.beam_width:
                _logger.debug(
                    "Pruning beam: %d → %d",
                    len(candidates),
                    self.beam_width,
                )

            frontier = candidates[: self.beam_width]

            _logger.debug(
                "New frontier (top scores): %s",
                [f"{n.score:.2f}" for n in frontier],
            )

        _logger.info("Max depth reached without finding goal")
        return None

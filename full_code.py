from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import copy
import math
import itertools
import networkx as nx


import logging
import math
import networkx as nx
from typing import Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

############################
# env.py — Sokoban Environment
############################

WALL = "#"
FLOOR = " "
PLAYER = "@"
BOX = "$"
TARGET = "."
BOX_ON_TARGET = "*"
PLAYER_ON_TARGET = "+"

ACTIONS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


@dataclass(frozen=True)
class SokobanState:
    grid: Tuple[Tuple[str, ...], ...]
    player_pos: Tuple[int, int]

    def is_goal(self) -> bool:
        for row in self.grid:
            for cell in row:
                if cell == TARGET or cell == PLAYER_ON_TARGET:
                    return False
        return True

    def render(self) -> str:
        return "\n".join("".join(row) for row in self.grid)

    def step(self, action: str) -> Optional["SokobanState"]:
        dx, dy = ACTIONS[action]
        x, y = self.player_pos
        nx, ny = x + dx, y + dy

        def at(pos):
            return self.grid[pos[0]][pos[1]]

        def set_cell(grid, pos, value):
            grid[pos[0]][pos[1]] = value

        if at((nx, ny)) in [WALL]:
            return None

        new_grid = [list(row) for row in self.grid]

        # Player moves into empty / target
        if at((nx, ny)) in [FLOOR, TARGET]:
            set_cell(new_grid, (x, y),
                     TARGET if at((x, y)) == PLAYER_ON_TARGET else FLOOR)
            set_cell(new_grid, (nx, ny),
                     PLAYER_ON_TARGET if at((nx, ny)) == TARGET else PLAYER)

        # Player pushes box
        elif at((nx, ny)) in [BOX, BOX_ON_TARGET]:
            bx, by = nx + dx, ny + dy
            if at((bx, by)) not in [FLOOR, TARGET]:
                return None

            set_cell(new_grid, (bx, by),
                     BOX_ON_TARGET if at((bx, by)) == TARGET else BOX)
            set_cell(new_grid, (nx, ny),
                     PLAYER_ON_TARGET if at((nx, ny)) == BOX_ON_TARGET else PLAYER)
            set_cell(new_grid, (x, y),
                     TARGET if at((x, y)) == PLAYER_ON_TARGET else FLOOR)

        else:
            return None

        return SokobanState(
            grid=tuple(tuple(row) for row in new_grid),
            player_pos=(nx, ny),
        )


def parse_puzzle(text: str) -> SokobanState:
    lines = text.strip("\n").splitlines()
    grid = []
    player_pos = None

    for i, line in enumerate(lines):
        row = []
        for j, c in enumerate(line):
            if c == PLAYER:
                player_pos = (i, j)
            if c == PLAYER_ON_TARGET:
                player_pos = (i, j)
            row.append(c)
        grid.append(tuple(row))

    return SokobanState(tuple(grid), player_pos)


############################
# policy.py — One-step LLM Policy
############################

class OneStepPolicy:
    """
    Abstract interface — LLM MUST obey this contract.
    """
    def predict(self, state: SokobanState) -> List[Tuple[str, float]]:
        raise NotImplementedError


class MockHeuristicPolicy(OneStepPolicy):
    """
    Replace with a real LLM later.
    """
    def predict(self, state: SokobanState) -> List[Tuple[str, float]]:
        # Uniform prior (baseline)
        return [(a, 1.0) for a in ACTIONS.keys()]


############################
# search.py — Beam Search
############################

@dataclass
class SearchNode:
    state: SokobanState
    parent: Optional["SearchNode"]
    action: Optional[str]
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
        beam_width: int = 5,
        max_depth: int = 50,
    ):
        self.policy = policy
        self.beam_width = beam_width
        self.max_depth = max_depth

        # Search tree for visualization / debugging
        self.graph = nx.DiGraph()

    def solve(self, start: SokobanState) -> Optional[SearchNode]:
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

        logger.info("Starting beam search")
        logger.info("Beam width = %d | Max depth = %d", self.beam_width, self.max_depth)

        # Main beam search loop (depth-limited)
        for depth in range(self.max_depth):
            logger.debug("=" * 40)
            logger.debug("Depth %d | Frontier size: %d", depth, len(frontier))

            candidates = []  # all children from this depth

            # Expand every node currently in the beam
            for node in frontier:
                logger.debug(
                    "Expanding node %d | depth=%d | score=%.3f",
                    node.id,
                    node.depth,
                    node.score,
                )

                # Check goal BEFORE expanding
                if node.state.is_goal():
                    logger.info("Goal reached at depth %d", node.depth)
                    return node

                # Hashable key for visited check
                state_key = node.state.grid

                # If we've already expanded this state, skip it
                if state_key in visited:
                    logger.debug(
                        "Skipping node %d (state already visited)", node.id
                    )
                    continue

                visited.add(state_key)

                # Query the one-step policy (LLM)
                action_scores = self.policy.predict(node.state)
                logger.debug(
                    "Policy returned %d candidate actions", len(action_scores)
                )

                valid_children = 0

                # Try all actions suggested by the policy
                for action, logit in action_scores:
                    next_state = node.state.step(action)

                    # Invalid move (wall, blocked push, etc.)
                    if next_state is None:
                        logger.debug(
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
                    logger.debug(
                        "Node %d is a dead end (no valid actions)", node.id
                    )
                else:
                    logger.debug(
                        "Node %d produced %d valid children",
                        node.id,
                        valid_children,
                    )

            # If nothing to expand further, terminate early
            if not candidates:
                logger.warning(
                    "Search terminated early at depth %d (no candidates)", depth
                )
                return None

            # Sort all candidates by score (higher = better)
            candidates.sort(key=lambda n: n.score, reverse=True)

            # Prune to beam width
            if len(candidates) > self.beam_width:
                logger.debug(
                    "Pruning beam: %d → %d",
                    len(candidates),
                    self.beam_width,
                )

            frontier = candidates[: self.beam_width]

            logger.debug(
                "New frontier (top scores): %s",
                [f"{n.score:.2f}" for n in frontier],
            )

        logger.info("Max depth reached without finding goal")
        return None



############################
# visualize.py
############################

def extract_plan(goal_node: SearchNode) -> List[str]:
    actions = []
    node = goal_node
    while node.parent is not None:
        actions.append(node.action)
        node = node.parent
    return list(reversed(actions))


def render_plan(start: SokobanState, plan: List[str]):
    state = start
    print("Initial State:")
    print(state.render())
    print()

    for i, action in enumerate(plan):
        state = state.step(action)
        print(f"Step {i+1}: {action}")
        print(state.render())
        print()


def export_search_graph(graph: nx.DiGraph, path="search_graph.dot"):
    nx.drawing.nx_pydot.write_dot(graph, path)


############################
# main.py — Example Run
############################

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    puzzle = """
#####
# @ #
# $ #
#   #
# . #
#####
"""

    state = parse_puzzle(puzzle)
    print(state.render())
    policy = MockHeuristicPolicy()
    solver = BeamSearchSolver(policy, beam_width=10)

    goal = solver.solve(state)

    if goal:
        plan = extract_plan(goal)
        render_plan(state, plan)
        export_search_graph(solver.graph)
        print("Solved with plan:", plan)
    else:
        print("No solution found.")

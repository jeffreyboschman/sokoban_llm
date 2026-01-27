import logging

from src.sokoban import parse_puzzle
from src.tools.visualize import extract_plan, render_plan, export_search_graph
from src.search.beamsearch import BeamSearchSolver
from src.policy import MockHeuristicPolicy


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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

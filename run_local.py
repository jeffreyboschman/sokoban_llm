from src.logging import setup_logging

from src.sokoban import parse_puzzle
from src.tools.visualize import extract_plan, render_plan, export_search_graph
from src.search.beamsearch import BeamSearchSolver
from src.policy.llm_policy import LLMOneStepPolicy, MistralOneStepPolicy



if __name__ == "__main__":
    setup_logging()

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
    policy = MistralOneStepPolicy()
    solver = BeamSearchSolver(policy, beam_width=10)

    goal = solver.solve(state)

    if goal:
        plan = extract_plan(goal)
        render_plan(state, plan)
        export_search_graph(solver.graph)
        print("Solved with plan:", plan)
    else:
        print("No solution found.")

import time
import argparse
import logging
from src.logging import setup_logging
from src.sokoban import parse_puzzle
from src.tools.visualize import extract_plan
from src.search.beamsearch import BeamSearchSolver
from src.policy.llm_policy import MistralOneStepPolicy
from src.policy.baseline_policy import MockHeuristicPolicy

# List of puzzle strings (selected from Microban levels 1-10)
puzzles = {
    1: """
####
# .#
#  ###
#*@  #
#  $ #
#  ###
####
""",
    2: """
######
#    #
# #@ #
# $* #
# .* #
#    #
######
""",
    3: """
  ####
###  ####
#     $ #
# #  #$ #
# . .#@ #
#########
""",
    4: """
########
#      #
# .**$@#
#      #
#####  #
    ####
""",
    70: """
#####
# @ ####
#      #
# $ $$ #
##$##  #
#   ####
# ..  #
##..  #
 ###  #
   ####
""",
    71: """
###########
#     #   ###
# $@$ # .  .#
# ## ### ## #
# #       # #
# #   #   # #
# ######### #
#           #
#############
""",
    72: """
  ####
 ##  #####
 #  $  @ #
 #  $#   #
#### #####
#  #   #
#    $ #
# ..#  #
#  .####
#  ##
####
""",
    153: """
#############
#.# @#  #   #
#.#$$   # $ #
#.#  # $#   #
#.# $#  # $##
#.#  # $#  #
#.# $#  # $#
#..  # $   #
#..  #  #  #
############
""",
    154: """
 ############################
 #                          #
 # ######################## #
 # #                      # #
 # # #################### # #
 # # #                  # # #
 # # # ################ # # #
 # # # #              # # # #
 # # # # ############ # # # #
 # # # # #            # # # #
 # # # # # ############ # # #
 # # # # #              # # #
 # # # # ################ # #
 # # # #                  # #
##$# # #################### #
#. @ #                      #
#############################
""",
    155: """
    ######               ####
#####*#  #################  ##
#   ###                      #
#        ########  ####  ##  #
### ####     #  ####  ####  ##
#*# # .# # # #     #     #   #
#*# #  #     # ##  # ##  ##  #
###    ### ###  # ##  # ##  ##
 #   # #*#      #     # #    #
 #   # ###  #####  #### #    #
 #####   #####  ####### ######
 #   # # #**#               #
## # #   #**#  #######  ##  #
#    #########  #    ##### ###
# #             # $        #*#
#   #########  ### @#####  #*#
#####       #### ####   ######
"""
}


def main(args: argparse.Namespace):
    if args.policy == 'mistral':
        policy = MistralOneStepPolicy()
    else:
        policy = MockHeuristicPolicy()

    solver = BeamSearchSolver(policy, beam_width=args.beam_width)

    logger.info(f"Running batch test with policy: {args.policy}, beam_width: {args.beam_width}")

    results = []

    for i, puzzle_text in list(puzzles.items())[:args.num_puzzles]:
        logger.info(f"\nTesting Puzzle {i}...")
        state = parse_puzzle(puzzle_text)
        if args.verbose:
            logger.info("Initial state:")
            logger.info(state.render())

        start_time = time.time()
        goal = solver.solve(state)
        end_time = time.time()
        duration = end_time - start_time

        solved = goal is not None
        plan_length = len(extract_plan(goal)) if solved else 0
        nodes_expanded = len(solver.graph) if hasattr(solver, 'graph') else 0

        result = {
            'puzzle': i,
            'solved': solved,
            'time': duration,
            'plan_length': plan_length,
            'nodes_expanded': nodes_expanded
        }
        results.append(result)

        logger.info(f"Result: Solved={solved}, Time={duration:.2f}s, Plan Length={plan_length}, Nodes Expanded={nodes_expanded}")

        if solved and args.verbose:
            plan = extract_plan(goal)
            logger.info(f"Plan: {plan}")
        elif not solved:
            logger.info("No solution found.")

    # Summary
    logger.info("\n" + "="*50)
    logger.info("SUMMARY")
    logger.info("="*50)
    total_solved = sum(1 for r in results if r['solved'])
    total_time = sum(r['time'] for r in results)
    avg_time = total_time / len(results)
    logger.info(f"Total Puzzles: {len(results)}")
    logger.info(f"Solved: {total_solved}")
    logger.info(f"Success Rate: {total_solved / len(results) * 100:.1f}%")
    logger.info(f"Total Time: {total_time:.2f}s")
    logger.info(f"Average Time: {avg_time:.2f}s")

    # Save results to file
    with open(args.output_file, 'w') as f:
        f.write("Puzzle,Solved,Time(s),Plan Length,Nodes Expanded\n")
        for r in results:
            f.write(f"{r['puzzle']},{r['solved']},{r['time']:.2f},{r['plan_length']},{r['nodes_expanded']}\n")
        f.write(f"\nSummary: Solved {total_solved}/{len(results)}, Total Time {total_time:.2f}s, Avg Time {avg_time:.2f}s\n")

    logger.info(f"Results saved to {args.output_file}")

def parse_args():
    parser = argparse.ArgumentParser(description="Batch test Sokoban puzzles")
    parser.add_argument('--policy', choices=['mistral', 'baseline'], default='baseline', help='Policy to use for solving')
    parser.add_argument('--beam_width', type=int, default=2, help='Beam width for the search')
    parser.add_argument('--num_puzzles', type=int, default=len(puzzles), help='Number of puzzles to test')
    parser.add_argument('--output_file', type=str, default='batch_test_results.txt', help='Output file for results')
    parser.add_argument('--log_level', type=str, default='INFO', help='Logging level (DEBUG, INFO, WARNING, ERROR)')
    parser.add_argument('--verbose', action='store_true', help='Print detailed output including plans and initial states')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    main(args)
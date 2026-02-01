import time
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

if __name__ == "__main__":
    setup_logging()

    policy = MockHeuristicPolicy()
    solver = BeamSearchSolver(policy, beam_width=4)

    results = []

    for i, puzzle_text in puzzles.items():
        print(f"\nTesting Puzzle {i}...")
        state = parse_puzzle(puzzle_text)
        print("Initial state:")
        print(state.render())

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

        print(f"Result: Solved={solved}, Time={duration:.2f}s, Plan Length={plan_length}, Nodes Expanded={nodes_expanded}")

        if solved:
            plan = extract_plan(goal)
            print("Plan:", plan)
        else:
            print("No solution found.")

    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    total_solved = sum(1 for r in results if r['solved'])
    total_time = sum(r['time'] for r in results)
    avg_time = total_time / len(results)
    print(f"Total Puzzles: {len(results)}")
    print(f"Solved: {total_solved}")
    print(f"Success Rate: {total_solved / len(results) * 100:.1f}%")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Average Time: {avg_time:.2f}s")

    # Save results to file
    with open('batch_test_results.txt', 'w') as f:
        f.write("Puzzle,Solved,Time(s),Plan Length,Nodes Expanded\n")
        for r in results:
            f.write(f"{r['puzzle']},{r['solved']},{r['time']:.2f},{r['plan_length']},{r['nodes_expanded']}\n")
        f.write(f"\nSummary: Solved {total_solved}/{len(results)}, Total Time {total_time:.2f}s, Avg Time {avg_time:.2f}s\n")

    print("Results saved to batch_test_results.txt")
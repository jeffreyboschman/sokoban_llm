
from src.sokoban import SokobanState
from src.search.beamsearch import SearchNode

import networkx as nx

def extract_plan(goal_node: SearchNode) -> list[str]:
    actions = []
    node = goal_node
    while node.parent is not None:
        actions.append(node.action)
        node = node.parent
    return list(reversed(actions))


def render_plan(start: SokobanState, plan: list[str]):
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
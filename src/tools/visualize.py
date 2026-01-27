
from src.sokoban import SokobanState
from src.search.beamsearch import SearchNode

import networkx as nx

import logging

_logger = logging.getLogger(__name__)

def extract_plan(goal_node: SearchNode) -> list[str]:
    actions = []
    node = goal_node
    while node.parent is not None:
        actions.append(node.action)
        node = node.parent
    return list(reversed(actions))


def render_plan(start: SokobanState, plan: list[str]):
    state = start
    _logger.info("Initial State:\n%s", state.render())

    for i, action in enumerate(plan):
        state = state.step(action)
        _logger.info("Step %d: %s \n%s", i+1, action, state.render())


def export_search_graph(graph: nx.DiGraph, path="search_graph.dot"):
    nx.drawing.nx_pydot.write_dot(graph, path)
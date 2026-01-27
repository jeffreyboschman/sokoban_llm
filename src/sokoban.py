from __future__ import annotations
from dataclasses import dataclass


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
    grid: tuple[tuple[str, ...], ...]
    player_pos: tuple[int, int]

    def is_goal(self) -> bool:
        for row in self.grid:
            for cell in row:
                if cell == TARGET or cell == PLAYER_ON_TARGET:
                    return False
        return True

    def render(self) -> str:
        return "\n".join("".join(row) for row in self.grid)

    def step(self, action: str) -> SokobanState | None:
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

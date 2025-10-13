"""Solving the sliding puzzle (or 8-puzzle) with local search. You are going to
implement the heuristics seen at the lecture, hill climbing, and tabu search.
The states are row-major flattened versions of the puzzle.

The strategy I recommend is to implement the simplest heuristic (# of misplaced
tiles) and the simpler search algorithm (hill climbing) first, check that they
work on easier puzzles, and continue with the rest of the heuristics and tabu
search.

You only need to modify the code in the "YOUR CODE HERE" sections. """

import random
from functools import partial

from typing import Callable, Generator, Optional, Any

import FreeSimpleGUI as sg #type: ignore

from framework.gui import BoardGUI
from framework.board import Board

BLANK_IMAGE_PATH = 'tiles/chess_blank_scaled.png'
sg.set_options(scaling=2)

"""The state is a tuple with 9 integers. For convenience we just define it as a
tuple of integers."""
State = tuple[int, ...]

goal: State = (1, 2, 3, 8, 0, 4, 7, 6, 5)


class SlidingBoard(Board):
    def __init__(self, start: State):
        self.m = 3
        self.n = 3
        self.create_board()
        self.update_from_state(start)

    def update_from_state(self, state: State) -> None:
        """Updates the board from the state of the puzzle."""
        for i, field in enumerate(state):
            self.board[i // self.n][i % self.n] = field

    def _default_state_for_coordinates(self, i: int, j: int) -> int:
        return 0


class SlidingProblem:
    """The search problem for the sliding puzzle."""

    def __init__(self, start_permutations: int = 10):
        self.goal : State = goal
        self.nil : State = (0,) * 9
        self.possible_slides = (
            (1, 3),         # from the upper left corner, you can move to right (+1) or down (+3)
            (-1, 1, 3),     # from the upper middle tile, you can move to left (-1), right (+1) or down (+3)
            (-1, 3),        # ...
            (-3, 1, 3),
            (-1, 1, -3, 3),
            (-1, -3, 3),
            (1, -3),
            (-1, 1, -3),
            (-1, -3),
        )
        self.start : State = self.generate_start_state(start_permutations)

    def start_state(self) -> State:
        return self.start

    def next_states(self, state: State) -> set[State]:
        ns = set()
        empty_ind = state.index(0)
        slides = self.possible_slides[empty_ind]
        for s in slides:
            ns.add(self.switch(state, empty_ind, empty_ind + s))
        return ns

    def is_goal_state(self, state: State) -> bool:
        return state == self.goal

    def generate_start_state(self, num_permutations: int) -> State:
        start = self.goal
        for _ in range(num_permutations):
            empty_ind = start.index(0)
            slides = self.possible_slides[empty_ind]
            start = self.switch(start, empty_ind, empty_ind + random.choice(slides))
        return start

    def switch(self, current: State, first: int, second: int) -> State:
        new = list(current)
        new[first], new[second] = new[second], new[first]
        return tuple(new)

HeuristicFunction = Callable[[State], int]
Algorithm = Callable[[SlidingProblem, HeuristicFunction], Generator]

# YOUR CODE HERE

# search


def hill_climbing(
    problem: SlidingProblem, f: HeuristicFunction
) -> Generator[State, None, None]:
    """The hill climbing search algorithm.

    Parameters
    ----------

    problem : SlidingProblem
      The search problem
    f : HeuristicFunction
      The heuristic function that evaluates states. Its input is a state.
    """
    current = problem.start_state()
    parent = problem.nil
    while not problem.is_goal_state(current):
        yield current #yielding each state
        next_states = problem.next_states(current)
        # TODO:
        # if with three branches
        # Hint: pseudocode from lecture 3 (local search), slide 5
        #       return None if no solution can be found

        # First branch
        if not next_states: return None
        
        # Removing children (gamma - pi)
        next_states_without_parent = set(next_states)
        if parent != problem.nil and parent in next_states_without_parent:
            next_states_without_parent.remove(parent)

        # Second branch
        if not next_states_without_parent:
            if parent == problem.nil: return
            current = parent
            parent = problem.nil
        # Third branch
        else:
            chosen = min(next_states_without_parent, key = f)
            parent = current
            current = chosen


    yield current


def tabu_search(
    problem: SlidingProblem,
    f: HeuristicFunction,
    tabu_len: int = 10,
    long_time: int = 1000,
) -> Generator[State, None, None]:
    """The tabu search algorithm.

    Parameters
    ----------

    problem : SlidingProblem
      The search problem
    f : HeuristicFunction
      The heuristic function that evaluates states. Its input is a state.
    tabu_len : int
      The length of the tabu list.
    long_time : int
      If the optimum has not changed in 'long_time' steps, the algorithm stops.
    """
    
    # Hint: pseudocode from lecture 3 (local search), slide 11
    #       return None if no solution is found
    #       don't forget to yield each state
    #       don't forget about set operations (such as subtraction)

    current = problem.start_state()
    opt = problem.start_state()
    tabu: list[State] = []
    since_last_improvement = 0

    #while not (problem.is_goal_state(current) or since_last_improvement >= long_time):   # Wouldn't show the last state (goal)
    while True:
        yield current
        if problem.is_goal_state(current) or since_last_improvement >= long_time: return   # Exit condition instead in case it's because of the while quitting early?
        next_states = problem.next_states(current)

        # First branch
        if not next_states: return None

        # Removing tabu (gamma - tabu)
        next_states_without_tabu = [s for s in next_states if s not in tabu]

        # Second branch
        if not next_states_without_tabu:
            chosen = min(next_states, key = f)
        #Third branch
        else:
            chosen = min(next_states_without_tabu, key = f)

        # Update tabu
        tabu.append(chosen)
        if len(tabu) > tabu_len: tabu.pop(0)

        current = chosen

        # Update opt
        if f(current) < f(opt):
            opt = current
            since_last_improvement = 0
        else:
            since_last_improvement += 1


# heuristics


def misplaced(state: State) -> int:
    # Hint: description on lecture 3 (local search) slide 22
    count = 0
    for i in range(9):
        tile = state[i]

        if tile != 0 and tile != goal[i]:
            count += 1
    
    return count


# Precomputed goal row/column indices,
# so finding them is not needed on every single Manhattan check
goal_state_rows = [0] * 9
goal_state_cols = [0] * 9
for i in range(9):
    tile = goal[i]
    goal_state_rows[tile] = i // 3      # div without remainder gives row nr
    goal_state_cols[tile] = i % 3    # remainder gives column nr

def manhattan(state: State) -> int:
    # Hint: description on lecture 3 (local search) slide 22
    total_dist = 0

    for i in range(9):
        tile = state[i]

        # no distance calc since it's not really a tile
        if tile == 0: continue

        current_row = i // 3
        current_col = i % 3

        target_row = goal_state_rows[tile]
        target_col = goal_state_cols[tile]

        row_dist = abs(current_row - target_row)
        col_dist = abs(current_col - target_col)

        total_dist += (row_dist + col_dist)

    return total_dist


# END OF YOUR CODE

start_permutations = 10

sliding_draw_dict = {
    i: (f"{i}", ("black", "lightgrey"), BLANK_IMAGE_PATH) for i in range(1, 9)
}
sliding_draw_dict.update({0: (" ", ("black", "white"), BLANK_IMAGE_PATH)})

sliding_problem = SlidingProblem(start_permutations)
board = SlidingBoard(sliding_problem.start)
board_gui = BoardGUI(board, sliding_draw_dict)

algorithms : dict[str, Algorithm] = {"Hill climbing": hill_climbing, "Tabu search": tabu_search}

heuristics : dict[str, HeuristicFunction] = {"Misplaced": misplaced, "Manhattan": manhattan,}

layout = [
    [
        sg.Column(board_gui.board_layout),
        sg.Frame("Log", [[sg.Output(size=(30, 10), key="log")]]),
    ],
    [
        sg.Frame(
            "Algorithm settings",
            [
                [
                    sg.T("Algorithm: "),
                    sg.Combo(
                        [algo for algo in algorithms], key="algorithm", readonly=True, default_value="Hill climbing"
                    ),
                    sg.T("Tabu length:"),
                    sg.Spin(
                        values=list(range(1000)),
                        initial_value=10,
                        key="tabu_len",
                        size=(5, 1),
                    ),
                ],
                [
                    sg.T("Heuristics: "),
                    sg.Combo(
                        [heur for heur in heuristics], key="heuristics", readonly=True, default_value="Misplaced"
                    ),
                ],
                [sg.Button("Change", key="Change_algo")],
            ],
        ),
        sg.Frame(
            "Problem settings",
            [
                [
                    sg.T("Starting permutations: "),
                    sg.Spin(
                        values=list(range(1, 100)),
                        initial_value=start_permutations,
                        key="start_permutations",
                        size=(5, 1),
                    ),
                ],
                [sg.Button("Change", key="Change_problem")],
            ],
        ),
    ],
    [sg.T("Steps: "), sg.T("0", key="steps", size=(7, 1), justification="right")],
    [sg.Button("Restart"), sg.Button("Step"), sg.Button("Go!"), sg.Button("Exit")],
]

window = sg.Window(
    "Sliding puzzle problem", layout, default_button_element_size=(10, 1), location=(0,0), finalize=True
)

starting = True
go = False
steps = 0

while True:  # Event Loop
    event, values = window.Read(0)
    window.Element("tabu_len").Update(disabled=values["algorithm"] != "Tabu search")
    window.Element("Go!").Update(text="Stop!" if go else "Go!")
    if event is None or event == "Exit" or event == sg.WIN_CLOSED:
        break
    if event == "Change_algo" or event == "Change_problem" or starting:
        if event == "Change_problem":
            start_permutations = int(values["start_permutations"])
            sliding_problem = SlidingProblem(start_permutations)
        algorithm : Any = algorithms[values["algorithm"]]
        heuristic = heuristics[values["heuristics"]]
        if algorithm is tabu_search:
            tabu_len = int(values["tabu_len"])
            algorithm = partial(algorithm, tabu_len=tabu_len)
        algorithm = partial(algorithm, f=heuristic)
        path = algorithm(sliding_problem)
        steps = 0
        window.Element("log").Update("")
        starting = False
        stepping = True
    if event == "Restart":
        path = algorithm(sliding_problem)
        steps = 0
        window.Element("log").Update("")
        stepping = True
    if event == "Step" or go or stepping:
        try:
            state = next(path)
        except StopIteration:
            pass
        else:
            print(f"{state}: {heuristic(state)}")
            window.Element("steps").Update(f"{steps}")
            steps += 1
        board.update_from_state(state)
        board_gui.update()
        stepping = False
    if event == "Go!":
        go = not go

window.Close()

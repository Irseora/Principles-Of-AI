from typing import Callable, Optional
from random import choice, randint, shuffle, choices, random
from enum import Flag
import FreeSimpleGUI as sg  # type: ignore

QUEEN_IMAGE_PATH = "tiles/queen_scaled.png"
BLANK_IMAGE_PATH = "tiles/chess_blank_scaled.png"

sg.change_look_and_feel("SystemDefault")

##################################################
################### STATE TYPE ###################
##################################################

State = list[int] #list of 8 integers

def valid_state(state : State) -> bool:
    '''Checks if `state` is a list of nonnegative integers below 8'''
    if not isinstance(state, list):
        return False
    return (
        len(set(state)) == len(state) and
        all(0 <= num < 8 and isinstance(num, int) for num in state)
    )

def valid_population(states : list[State]) -> bool:
    '''Checks if every element of `states` is a valid state'''
    return all(valid_state(state) for state in states)

##################################################
########## INITIALIZATION OF POPULATION ##########
##################################################

def random_state() -> State:
    '''Returns a randomly initizalized state'''
    state = list(range(8))
    shuffle(state)
    return state

def random_population(population_size : int) -> list[State]:
    '''Returns a randomly initizalized population (list of states)'''
    return [ random_state() for _ in range(population_size) ]

##################################################
################ FITNESS FUNCTION ################
##################################################

def fitness(state : State) -> int:
    '''Returns the number of nonattacking pairs of queens'''
    assert valid_state(state)
    attacks = 0
    for x_a, y_a in enumerate(state):
        for x_b, y_b in enumerate(state):
            if x_b < x_a+1: continue
            attacks += int(x_a+y_a == x_b+y_b) + int(y_a-x_a == y_b-x_b)
    return sum(range(8)) - attacks

def is_solution(state : State) -> bool:
    '''Returns if state is a goal state (# nonatacking queen pairs == # queen pairs'''
    return fitness(state) == sum(range(8))

def contains_solution(states : list[State]) -> Optional[State]:
    '''Returns the solution state (for visualization) if there is any in the population'''
    for state in states:
        if is_solution(state):
            return state
    return None

##################################################
############### PRINTING FUNCTIONS ###############
##################################################

def print_state(state : State) -> None:
    queens = list(enumerate(state))
    for i in range(8):
        for j in range(8):
            print('|', end='')
            if (i,j) in queens:
                print('Q', end='')
            else:
                print('_', end='')
        print('|')


def print_population(states : list[State], f : Callable[[State],int] = fitness) -> None:
    for state in states:
        assert valid_state(state)
        print(state, '-->', f(state))
    print('#'*31)

##################################################
################### SELECTION ####################
##################################################

def selection(states : list[State], min_val : int, 
                f : Callable[[State],int] = fitness, 
                oversampling : bool = True
             ) -> list[State]:
    '''Applies regular selection on population'''
    preserved_states = [ state for state in states if f(state) > min_val ]
    if oversampling:
        while len(preserved_states) < len(states):
            preserved_states.append(choice(preserved_states))
    assert valid_population(preserved_states)
    return preserved_states

def selection_roulette(states : list[State], 
                        f : Callable[[State],int] = fitness
                      ) -> list[State]:
    '''Applies roulette wheel selection on population'''
    pass # TODO
    # Hint: Check lecture 7 (evolution) slide 14
    #       Use choices from random library (and the weights and k optional argument)
    #           to choose k "good" states randomly
    #       The weights should be the deduced using the fitness function
    #       Finally, return the preserved states.


##################################################
################# RECOMBINATION ##################
##################################################

def recombination(states : list[State]) -> list[State]:
    '''Applies recombination step on population'''
    assert isinstance(states,list)
    assert valid_population(states)
    new_states : list[State] = []
    for i in range(0, len(states)-1, 2):
        new_states += recombine(states[i], states[i+1])
    return new_states

def recombine(state_a : State, state_b : State) -> tuple[State, State]:
    '''Applies recombination step on two states'''
    pass # TODO
    # Hint: Check lecture 7 (evolution) slide 12
    #       choose 2 random indexes for division barriers, and then 
    #       perform the recombination

##################################################
##################### REPAIR #####################
##################################################

def repair(states : list[State]) -> list[State]:
    '''Applies repair step on population'''
    new_states : list[State] = []
    for i in range(0, len(states)-1, 2):
        new_states += repair_states(states[i], states[i+1])
    assert valid_population(new_states)
    return new_states

def repair_states(state_a : State, state_b : State) -> tuple[State,State]:
    '''Applies repair step on two states'''
    state_a_, state_b_ = state_a.copy(), state_b.copy()
    pass # TODO
    # Hint: Check for each element in state_a_ if it is contained twice.
    #       If so, find a good substitute for it in state_b_
    return (state_a_, state_b_)

##################################################
#################### MUTATION ####################
##################################################

def mutation(states : list[State], chance : float) -> list[State]:
    '''Applies mutation step on population'''
    new_population = [mutate(state, chance) for state in states]
    assert valid_population(new_population)
    return new_population

def mutate(state : State, chance : float) -> State:
    '''Applies mutation step on one state'''
    mutated_state : State = state.copy()
    pass # TODO
    # Hint: Pick 2 random indexes and swap the elements on those positions
    #       Do that only with the given chance! (Suggestion: use random.random())

    return mutated_state

##################################################
################## REPLACEMENT ###################
##################################################

def replacement(original : list[State], evolved : list[State], 
                k : int,
                f : Callable[[State],int] = fitness
             ) -> list[State]:
    '''Applies replacement step based on original and evolved populations'''
    return sorted(original,key=f)[k:] + sorted(evolved,key=f)[-k:]

##################################################
###################### GUI #######################
##################################################

SquareState = Flag("SquareState", "W B Q U")

class Square:
    def __init__(self, initial_state: SquareState):
        self.state = initial_state

    def has_queen(self) -> bool:
        return bool(self.state & SquareState.Q)

    def set_queen(self):
        self.state = self.state | SquareState.Q

    def clear_queen(self):
        self.state = self.state & ~SquareState.Q

queens_draw_dict = {
    SquareState.W: ("", ("black", "white"), BLANK_IMAGE_PATH),
    SquareState.B: ("", ("black", "lightgrey"), BLANK_IMAGE_PATH),
    SquareState.W | SquareState.Q: ("", ("black", "white"), QUEEN_IMAGE_PATH),
    SquareState.B | SquareState.Q: ("", ("black", "lightgrey"), QUEEN_IMAGE_PATH),
}

class BoardGUI:
    def __init__(self):
        self.board_size = 8
        self.board_layout = []
        self.create()

    def create(self):
        self.board_layout = []
        for i in range(self.board_size):
            row = []
            for j in range(self.board_size):
                color = SquareState.W if (i + j) % 2 == 0 else SquareState.B
                square = Square(color)
                text, colors, image = queens_draw_dict[square.state]
                row.append(sg.Button(text, 
                                   key=(i, j),
                                   button_color=colors,
                                   image_filename=image,
                                   border_width=0,
                                   disabled=True))
            self.board_layout.append(row)

    def update_from_state(self, state: State, window):
        for i in range(self.board_size):
            for j in range(self.board_size):
                color = SquareState.W if (i + j) % 2 == 0 else SquareState.B
                if state[j] == i:
                    color = color | SquareState.Q
                text, colors, image = queens_draw_dict[color]
                window[(i, j)].update(
                    text=text, 
                    button_color=colors, 
                    image_filename=image,
                )
    def update_empty(self, window):
        self.update_from_state([-1]*8, window)

def create_window(board_gui):
    layout = [
        [
            sg.Column(board_gui.board_layout),
            sg.Column([
                [sg.Text("Population Log", font=("Helvetica", "16", "bold"))],
                [sg.Multiline(
                    size=(32, 20),
                    key="log",
                    autoscroll=True,
                    disabled=True,
                    font=("Courier", 18)
                )]
            ])
        ],
        [
            sg.Frame(
                "Algorithm settings",
                [
                    [
                        sg.T("Population size: ", size=(22, 1)),
                        sg.Input("4", key="pop_size", size=(15, 1)),
                    ],
                    [
                        sg.T("Max iterations: ", size=(22, 1)),
                        sg.Input("100", key="max_iter", size=(15, 1)),
                    ],
                    [
                        sg.T("Mutation prob: ", size=(22, 1)),
                        sg.Input("0.75", key="mut_prob", size=(15, 1)),
                    ],
                    [
                        sg.Checkbox("Use roulette selection", key="use_roulette", default=False, enable_events=True),
                    ],
                    [
                        sg.T("Min fitness (non-roulette): ", size=(22, 1)),
                        sg.Input("21", key="min_fitness", size=(15, 1), disabled=False),
                    ],
                ],
            ),
        ],
        [
            sg.T("Generation: "), sg.T("0", key="generation", size=(7, 1), justification="right"),
            sg.T("  Best fitness: "), sg.T("0", key="best_fitness", size=(7, 1), justification="right"),
            sg.T("  Status: "), sg.T("Not started", key="status", size=(30, 1)),
        ],
        [sg.Button("Restart"), sg.Button("Step"), sg.Button("Go!"), sg.Button("Exit")],
    ]

    window = sg.Window(
        "Evolutionary Algorithm - N-Queens",
        layout,
        default_button_element_size=(10, 1),
        auto_size_buttons=False,
        location=(0, 0),
        font=("Helvetica", "16"),
        scaling=0.8
    )
    return window

def log_population(window, population, generation):
    """Logs the current population to the log window"""
    log_text = f"=== Generation {generation} ===\n"
    for state in population:
        log_text += f"{state} --> {fitness(state)}\n"
    log_text += "#" * 31 + "\n\n"
    window["log"].print(log_text, end="")

def main():
    board_gui = BoardGUI()
    window = create_window(board_gui)
    window.Finalize()
    
    pop_size = 4
    population = random_population(pop_size)
    generation = 0
    go = False
    solution_found = False
    
    window["status"].update("Population initialized")
    log_population(window, population, generation)
    board_gui.update_empty(window)

    while True:
        event, values = window.Read(10 if go else None)
        
        if event is None or event == "Exit" or event == sg.WIN_CLOSED:
            break
        
        window.Element("Go!").Update(text="Stop!" if go else "Go!")
        
        if event == "use_roulette":
            window["min_fitness"].update(disabled=values["use_roulette"])
        
        if event == "Restart":
            try:
                pop_size = int(values["pop_size"])
                assert pop_size > 1
            except (ValueError, TypeError, AssertionError):
                window["status"].update("Invalid input values!")
            else:
                population = random_population(pop_size)
                generation = 0
                solution_found = False
                window["generation"].update("0")
                window["best_fitness"].update("0")
                window["status"].update("Population initialized")
                window["log"].update("")
                log_population(window, population, generation)
                board_gui.update_empty(window)
        
        if event == "Step" or (go and not solution_found):
            if population is None:
                window["status"].update("Click Restart to initialize")
                continue
            
            try:
                mutation_prob = float(values["mut_prob"])
                use_roulette = values["use_roulette"]
                min_fit = int(values["min_fitness"]) if not use_roulette else 21
                max_iter = int(values["max_iter"])
                assert 0 <= mutation_prob <= 1
                assert 28 >= min_fit >= 0
                assert max_iter > 0
            except (ValueError, TypeError, AssertionError):
                window["status"].update("Invalid parameter values!")
                go = False
            else:
                evolved_state = (
                    selection_roulette(population)
                    if use_roulette else
                    selection(population, min_fit)
                )
                shuffle(evolved_state)
                evolved_state = recombination(evolved_state)
                evolved_state = repair(evolved_state)
                evolved_state = mutation(evolved_state, mutation_prob)
                population = replacement(population, evolved_state, len(population)//2)
                
                generation += 1
                window["generation"].update(str(generation))
                
                log_population(window, population, generation)
                
                best_state = max(population, key=fitness)
                best_fit = fitness(best_state)
                window["best_fitness"].update(str(best_fit))
                
                solution = contains_solution(population)
                if solution:
                    solution_found = True
                    window["status"].update(f"Solution found!")
                    board_gui.update_from_state(solution, window)
                    go = False
                else:
                    window["status"].update(f"Running... (Gen {generation})")
                
                if generation >= max_iter and not solution_found:
                    window["status"].update("Max iterations reached - no solution")
                    go = False
        
        if event == "Go!":
            if population is None:
                window["status"].update("Click Restart to initialize")
            else:
                go = not go
                if go and solution_found:
                    go = False
    
    window.Close()

if __name__ == "__main__":
    main()
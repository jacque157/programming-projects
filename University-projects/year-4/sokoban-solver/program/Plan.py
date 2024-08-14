from Literals import *
from GameState import GameState
import tkinter as tk
#from Solve import Solver


class Plan:
    def __init__(self, map_path):
        self.initial_state = GameState.load_map(map_path)
        self.clauses = []
        
        self.tiles = self.init_tiles(self.initial_state)
        self.actions = self.init_actions(self.tiles)
        self.frame_problem_fluents = self.init_frame_problem_fluents(self.tiles)   

    def init_tiles(self, world):
        tiles = set()
        for row in range(len(world)):
            for col in range(len(world[row])):
                tiles.add(Plan.encode_tile(row, col))
        return tiles

    def init_frame_problem_fluents(self, tiles):
        fluents = set()
        for tile in tiles:
            fluents.add(At(tile))
            fluents.add(Box(tile))
        return fluents

    def init_actions(self, tiles):
        actions = set()
        actions.add(Wait())
        for tile0 in tiles:
            for tile1 in Plan.adjacent_tiles(tile0, tiles):
                actions.add(Move(tile0, tile1))

            for tile1, tile2 in Plan.two_adjacent_tiles(tile0, tiles):
                actions.add(Push(tile0, tile1, tile2))
            
        return actions

    def adjacent_tiles(tile, tiles):
        row, col = eval(tile)
        adjacent = Plan.encode_tile(row - 1, col)
        if adjacent in tiles:
            yield adjacent

        adjacent = Plan.encode_tile(row + 1, col)
        if adjacent in tiles:
            yield adjacent

        adjacent = Plan.encode_tile(row, col - 1)
        if adjacent in tiles:
            yield adjacent

        adjacent = Plan.encode_tile(row, col + 1)
        if adjacent in tiles:
            yield adjacent

    def two_adjacent_tiles(tile, tiles):
        row, col = eval(tile)
        adjacent1 = Plan.encode_tile(row - 1, col)
        adjacent2 = Plan.encode_tile(row - 2, col)
        if adjacent1 in tiles and adjacent2 in tiles:
            yield adjacent1, adjacent2

        adjacent1 = Plan.encode_tile(row + 1, col)
        adjacent2 = Plan.encode_tile(row + 2, col)
        if adjacent1 in tiles and adjacent2 in tiles:
            yield adjacent1, adjacent2

        adjacent1 = Plan.encode_tile(row, col - 1)
        adjacent2 = Plan.encode_tile(row, col - 2)
        if adjacent1 in tiles and adjacent2 in tiles:
            yield adjacent1, adjacent2

        adjacent1 = Plan.encode_tile(row, col + 1)
        adjacent2 = Plan.encode_tile(row, col + 2)
        if adjacent1 in tiles and adjacent2 in tiles:
            yield adjacent1, adjacent2

    def build_theory(self, steps):
        clauses = []
        
        clauses.append('c initial state')
        clauses.extend(self.encode_initial_state(self.initial_state))
        
        clauses.append('c goal state')
        clauses.extend(self.encode_goal_state(self.initial_state, steps))
        
        clauses.append('c push and move actions')
        clauses.extend(self.encode_actions(self.actions, steps))
        
        clauses.append('c frame problem')
        clauses.extend(self.encode_explanatory_frame_problem(self.frame_problem_fluents, self.actions, steps))
        
        clauses.append('c action exclusivity')
        clauses.extend(self.encode_action_exclusivity(self.actions, steps))
        
        return clauses
            
    def encode_initial_state(self, world):
        clauses = []

        for i, row in enumerate(world):
            for j, col in enumerate(row):
                tile = Plan.encode_tile(i, j)
                if col['wall']:
                    clauses.append(Clause([Wall(tile)]))
                else:
                    clauses.append(Clause([Wall(tile).negate()]))
                if col['goal']:
                    clauses.append(Clause([Goal(tile)]))
                else:
                    clauses.append(Clause([Goal(tile).negate()]))
                if col['box']:
                    clauses.append(Clause([Box(tile).at_step(0)]))
                else:
                    clauses.append(Clause([Box(tile).at_step(0).negate()]))
                if col['player']:
                    clauses.append(Clause([At(tile).at_step(0)]))
                else:
                    clauses.append(Clause([At(tile).at_step(0).negate()]))

        return clauses

    def encode_tile(i, j):
        return f'[{i},{j}]'
    
    def encode_goal_state(self, world, steps):
        clauses = []

        for i, row in enumerate(world):
            for j, col in enumerate(row):
                tile = Plan.encode_tile(i, j)
                clauses.append(Clause([Goal(tile).negate(), Box(tile).at_step(steps)]))

        return clauses

    def encode_actions(self, actions, steps):
        clauses = []
        for step in range(1, steps + 1):
            for action in actions:
                clauses.extend(self.encode_action(action, step))
        return clauses

    def encode_action(self, action, step):
        clauses = []
        for precondition in action.preconditions():
            if type(precondition) == Wall or type(precondition) == Goal:
                clauses.append(Clause([action.at_step(step).negate(), precondition]))
            else:
                clauses.append(Clause([action.at_step(step).negate(), precondition.at_step(step - 1)]))
        
        for effect in action.effects():
            if type(effect) == Wall or type(effect) == Goal:
                clauses.append(Clause([action.at_step(step).negate(), effect]))
            else:
                clauses.append(Clause([action.at_step(step).negate(), effect.at_step(step)]))

        return clauses

    def encode_frame_problem(self, fluents, actions, steps):
        clauses = []
        
        for action in actions:
            for fluent in fluents:             
                if type(fluent) == Wall or type(fluent) == Goal:
                    continue              
                if fluent not in action.effect_fluents():
                    for step in range(1, steps + 1):
                        clause = Clause([fluent.at_step(step - 1).negate(),
                                         action.at_step(step).negate(),
                                         fluent.at_step(step)])
                        clauses.append(clause)
                        clause = Clause([fluent.at_step(step - 1),
                                         action.at_step(step).negate(),
                                         fluent.at_step(step).negate()])
                        clauses.append(clause)
        return clauses

    def encode_explanatory_frame_problem(self, fluents, actions, steps):
        clauses = []

        for fluent in fluents:
            for step in range(1, steps + 1):
                clause1 = [fluent.at_step(step - 1), fluent.negate().at_step(step)] +\
                          [action.at_step(step) for action in actions
                           if fluent in action.effect_fluents()]
                clause2 = [fluent.negate().at_step(step - 1), fluent.at_step(step)] +\
                          [action.at_step(step) for action in actions
                           if fluent in action.effect_fluents()]

                clauses.append(Clause(clause1))
                clauses.append(Clause(clause2))
                
        return clauses
                        
    def encode_action_exclusivity(self, actions, steps):
        clauses = []
        for step in range(1, steps + 1):
            actions_at_step = [action.at_step(step) for action in actions]
            for i in range(len(actions_at_step)):
                for j in range(i + 1, len(actions_at_step)):
                    action1 = actions_at_step[i]
                    action2 = actions_at_step[j]
                    if action1 != action2:
                        at_most_one = [action1.negate(), action2.negate()] # -A \/ -B <-> -(A /\ B)
                        clauses.append(Clause(at_most_one))
            clauses.append(Clause(actions_at_step)) # A \/ B \/ C ...     
        return clauses
        

import tkinter as tk
from PIL import Image, ImageTk
from Constants import *
from AnimatedObject import *
from StaticObject import *


class GameState:
    OUT = -1
    EMPTY = 0
    BOX = 1
    WALL = 2

    def __init__(self, file, canvas):
        self.canvas = canvas
        self.player = None
        map_ = GameState.load_map(file)
        self.world = self.load_level(map_)
        self.canvas.config(width=(len(self.world[0])* TILE_WIDTH) + 2 * UNIT,
                           height=(len(self.world)* TILE_HEIGHT) + 2 * UNIT) 
        self.canvas.pack()

    def load_map(file):
        map_ = []
        with open(file) as f:
            for line in f:
                row = []
                for char in line:
                    elements = {'wall' : False, 'box' : False, 'goal' : False, 'player' : False}
                    if char == '\n':
                        continue
                 
                    if char == '#':
                        elements['wall'] = True
                    elif char == 'C':
                        elements['box'] = True
                    elif char == 'c':
                        elements['box'] = True
                        elements['goal'] = True
                    elif char == 'S':
                        elements['player'] = True
                    elif char == 's':
                        elements['player'] = True
                        elements['goal'] = True
                    elif char == 'X':
                        elements['goal'] = True                  
                    row.append(elements)
                    
                map_.append(row)
        return map_

    def load_level(self, map_):
        level = []
        for i, row in enumerate(map_):
            level_row = []
            for j, col in enumerate(row):
                elements = {'wall' : None, 'box' : None, 'goal' : None, 'player' : None}
                x = (TILE_WIDTH // 2) + (TILE_WIDTH * j)
                y = (TILE_HEIGHT // 2) + (TILE_HEIGHT * i)
                if col['wall']:
                    elements['wall'] = Wall(x, y , self.canvas)
                if col['box']:
                    elements['box'] = Box(x, y, self.canvas)
                if col['goal']:
                    elements['goal'] = Goal(x, y, self.canvas)
                if col['player']:
                    elements['player'] = Player(x, y, self.canvas)
                    self.player = elements['player']
                
                level_row.append(elements)
                
            level.append(level_row)
        return level
    
    def draw_map(self):
        self.canvas.delete('all')
        for row in self.world:
            for col in row:
                if col['goal']:
                    col['goal'].draw()
        
        for row in self.world:
            for col in row:
                
                if col['wall']:
                    col['wall'].draw()

                if col['box']:
                    col['box'].draw()
                if col['player']:                      
                    col['player'].draw()
                        
        self.canvas.update()

    def update_map(self):
        self.player.next_frame()
        self.player.next_position()
        for row in self.world:
            for col in row:
                if col['box']:
                    col['box'].next_frame()
                    col['box'].next_position()
                    col['box'].active = col['goal'] is not None
                 

    def obstacle_type(self, row, col):
        if row < 0 or row >= len(self.world) or col < 0 or col >= len(self.world[row]):
            return GameState.OUT
        if self.world[row][col]['wall']:
            return GameState.WALL
        if self.world[row][col]['box']:
            return GameState.BOX

        return GameState.EMPTY
    
    def next_tile(row, col, direction):
        if direction == 'Up':
            return row - 1, col
        if direction == 'Down':
            return row + 1, col
        if direction == 'Left':
            return row, col - 1
        if direction == 'Right':
            return row, col + 1
        return row, col
    
    def move_player(self, direction):
        row1, col1 = self.player.get_row_col()
        row2, col2 = GameState.next_tile(row1, col1, direction)
        if self.obstacle_type(row2, col2) == GameState.EMPTY:
            self.world[row1][col1]['player'] = None
            self.world[row2][col2]['player'] = self.player
            self.player.move(direction)

    def push_box(self, direction):
        row1, col1 = self.player.get_row_col()
        row2, col2 = GameState.next_tile(row1, col1, direction)
        row3, col3 = GameState.next_tile(row2, col2, direction)
        if self.obstacle_type(row2, col2) == GameState.BOX and \
           self.obstacle_type(row3, col3) == GameState.EMPTY:
            self.world[row1][col1]['player'] = None
            self.world[row2][col2]['player'] = self.player
            self.world[row3][col3]['box'] = self.world[row2][col2]['box']
            self.world[row2][col2]['box'] = None
            self.player.move(direction, push=True)
            self.world[row3][col3]['box'].move(direction)
        

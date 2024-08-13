import tkinter as tk
from PIL import Image, ImageTk
from Constants import *
from AnimatedObject import *
from StaticObject import *
from GameState import *
from tkinter import messagebox


class Game:

    def __init__(self, root, map_path):
        self.root = root
        self.canvas = tk.Canvas(root, bg='turquoise')
        self.game_state = GameState(map_path, self.canvas)
        self.active = True       

    def key_pressed(self, event):
        key = event.keysym
        if key == 'Up' or key == 'Down' or key == 'Right' or key == 'Left':     
            self.move(*self.game_state.player.get_row_col(), key)
       
    def move(self, row1, col1, direction):
        if self.game_state.player.is_idle():
            row2, col2 = GameState.next_tile(row1, col1, direction)
            if self.game_state.obstacle_type(row2, col2) == GameState.EMPTY:
                self.game_state.move_player(direction)
            elif self.game_state.obstacle_type(row2, col2) == GameState.BOX:
                self.game_state.push_box(direction)
        
    def main(self):
        self.canvas.bind_all('<Key>', self.key_pressed)
        while self.active:
            try:
                self.update()
                self.canvas.after(100)
            except:
                break

    def visualise_plan(self, sequence):
        while self.active:
            try:
                if self.game_state.player.is_idle():
                    if sequence:
                        direction = sequence.pop(0)
                        self.move(*self.game_state.player.get_row_col(), direction)
                self.update()
                self.canvas.after(100)
            except:
                break
            
    def update(self):
        self.game_state.update_map()
        self.game_state.draw_map()
        self.canvas.update()

                
if __name__ == '__main__':
    window = tk.Tk()
    window.title('Sokoban')
    g = Game(window, 'maps/map2.txt')
    g.main()
    #g.visualise_plan(['Right', 'Right', 'Right', 'Right', 'Right'])

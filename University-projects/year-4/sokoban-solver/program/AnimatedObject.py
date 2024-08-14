from Constants import *
import tkinter as tk
from PIL import Image, ImageTk


class AnimatedObject:
    X_OFFSET = UNIT
    Y_OFFSET = UNIT
        
    def __init__(self, x, y, canvas):
        self.x = x
        self.y = y
        self.new_x = x
        self.new_y = y
        self.dx = 0
        self.dy = 0
        self.canvas = canvas
        self.frame = 0
        self.images = []
        for path in self.image_paths():
            img = Image.open(path)
            img = img.resize((UNIT, UNIT), resample=Image.BOX)
            self.images.append(ImageTk.PhotoImage(img))
    
    def draw(self):
        self.canvas.create_image(AnimatedObject.X_OFFSET + self.x,
                                 AnimatedObject.Y_OFFSET + self.y,
                                 image=self.images[self.frame])
    def next_frame(self):
        self.frame = (self.frame + 1) % len(self.images)

    def next_position(self):
        self.x += self.dx
        self.y += self.dy
        if self.x == self.new_x and self.y == self.new_y:
            self.x = self.new_x
            self.y = self.new_y
            self.dx = 0
            self.dy = 0

    def move(self, x, y):        
        self.new_x = x
        self.new_y = y
        self.dx = (self.new_x - self.x) // 4
        self.dy = (self.new_y - self.y) // 3
        
    def image_paths(self):
        return ['']

class Player(AnimatedObject):
    MOVE_U = 0
    MOVE_D = 1
    MOVE_L = 2
    MOVE_R = 3
    
    PUSH_U = 4
    PUSH_D = 5
    PUSH_L = 6
    PUSH_R = 7

    IDLE_U = 8
    IDLE_D = 9
    IDLE_L = 10
    IDLE_R = 11
    
    def __init__(self, x, y, canvas):
        super().__init__(x, y, canvas)
        self.state = Player.IDLE_D
        self.frame = 1 + 3

    def image_paths(self):
        paths = []
        paths.extend([f'assets/move_U{i}.png' for i in range(1, 4)])
        paths.extend([f'assets/move_D{i}.png' for i in range(1, 4)])
        paths.extend([f'assets/move_L{i}.png' for i in range(1, 5)])
        paths.extend([f'assets/move_R{i}.png' for i in range(1, 5)])
        paths.extend([f'assets/push_U{i}.png' for i in range(1, 4)])
        paths.extend([f'assets/push_D{i}.png' for i in range(1, 4)])
        paths.extend([f'assets/push_L{i}.png' for i in range(1, 5)])
        paths.extend([f'assets/push_R{i}.png' for i in range(1, 5)])
        return paths

    def next_frame(self):
        if self.state == Player.IDLE_U:
            self.frame = 1
        elif self.state == Player.IDLE_D:
            self.frame = 4
        elif self.state == Player.IDLE_L:
            self.frame = 7
        elif self.state == Player.IDLE_R:
            self.frame = 11

        elif self.state == Player.MOVE_U:
            self.frame = 0 + ((self.frame + 1) % 3)
        elif self.state == Player.MOVE_D:
            self.frame = 3 + ((self.frame + 1) % 3)
        elif self.state == Player.MOVE_L:
            self.frame = 6 + ((self.frame + 1) % 4)
        elif self.state == Player.MOVE_R:
            self.frame = 10 + ((self.frame + 1) % 4)

        elif self.state == Player.PUSH_U:
            self.frame = 14 + ((self.frame + 1) % 3)
        elif self.state == Player.PUSH_D:
            self.frame = 17 + ((self.frame + 1) % 3)
        elif self.state == Player.PUSH_L:
            self.frame = 20 + ((self.frame + 1) % 4)
        elif self.state == Player.PUSH_R:
            self.frame = 24 + ((self.frame + 1) % 4)

    def is_idle(self):
        return self.state == Player.IDLE_U or self.state == Player.IDLE_D \
               or self.state == Player.IDLE_L or self.state == Player.IDLE_R

    def move(self, direction, push=False):
        x, y = self.x, self.y
        if direction == 'Up':
            y -= TILE_HEIGHT
            if push:
                self.frame = 14
                self.state = Player.PUSH_U 
            else:
                self.frame = 0
                self.state = Player.MOVE_U 
        elif direction == 'Down':
            y += TILE_HEIGHT
            if push:
                self.frame = 17
                self.state = Player.PUSH_D
            else:
                self.frame = 3
                self.state = Player.MOVE_D
        elif direction == 'Left':
            x -= TILE_WIDTH
            if push:
                self.frame = 20
                self.state = Player.PUSH_L
            else:
                self.frame = 6
                self.state = Player.MOVE_L
        elif direction == 'Right':
            x += TILE_WIDTH
            if push:
                self.frame = 24
                self.state = Player.PUSH_R
            else:
                self.frame = 10
                self.state = Player.MOVE_R
        super().move(x, y)

    def next_position(self):
        self.x += self.dx
        self.y += self.dy
        if self.x == self.new_x and self.y == self.new_y:
            self.dx = 0
            self.dy = 0
            if self.state == Player.MOVE_U or self.state == Player.PUSH_U:
                self.state = Player.IDLE_U
            elif self.state == Player.MOVE_D or self.state == Player.PUSH_D:
                self.state = Player.IDLE_D
            elif self.state == Player.MOVE_R or self.state == Player.PUSH_R:
                self.state = Player.IDLE_R
            elif self.state == Player.MOVE_L or self.state == Player.PUSH_L:
                self.state = Player.IDLE_L
            self.x = self.new_x
            self.y = self.new_y          
    
    def get_row_col(self):
        row = ((2 * self.y) - TILE_HEIGHT) // (2 * TILE_HEIGHT)
        col = ((2 * self.x) - TILE_WIDTH) // (2 * TILE_WIDTH)
        return row, col


class Box(AnimatedObject):
    def __init__(self, x, y, canvas):
        super().__init__(x, y, canvas)
        self.active = False

    def image_paths(self):
        return [f'assets/box{i // 2}.png' for i in range(2, 6)]

    def next_frame(self):
        if self.active:
            super().next_frame()
        else:
            self.frame = 0

    def move(self, direction, push=False):
        x, y = self.x, self.y
        if direction == 'Up':
            y -= TILE_HEIGHT
        elif direction == 'Down':
            y += TILE_HEIGHT
        elif direction == 'Left':
            x -= TILE_WIDTH
        elif direction == 'Right':
            x += TILE_WIDTH
        super().move(x, y)


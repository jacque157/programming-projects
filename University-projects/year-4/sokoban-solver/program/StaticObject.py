import tkinter as tk
from PIL import Image, ImageTk
from Constants import *


class StaticObject:
    X_OFFSET = UNIT
    Y_OFFSET = UNIT
        
    def __init__(self, x, y, canvas):
        self.x = x
        self.y = y
        self.canvas = canvas
        img = Image.open(self.image_path())
        img = img.resize((UNIT, UNIT), resample=Image.BOX)
        self.image = ImageTk.PhotoImage(img)
    
    def draw(self):
        self.canvas.create_image(StaticObject.X_OFFSET + self.x,
                                 StaticObject.Y_OFFSET + self.y,
                                 image=self.image)
    def image_path(self):
        return ''


class Wall(StaticObject):
    def image_path(self):
        return 'assets/wall.png'

class Goal(StaticObject):    
    def draw(self):
        x_offset = 1 * SCALE_FACTOR
        y_offset = 0 * SCALE_FACTOR
        self.canvas.create_image(StaticObject.X_OFFSET + self.x + x_offset,
                                 StaticObject.Y_OFFSET + self.y + y_offset,
                                 image=self.image)
    def image_path(self):
        return 'assets/x.png'

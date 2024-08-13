import numpy as np
import glm


class World:
    def __init__(self, path) -> None:
        self.level_map = self.read_map(path)
        self.ball = np.array((0, 1, 1), dtype=np.int16)
        self.direction = np.array((0, 1, 0), dtype=np.int16)
        self.down = np.array((-1, 0, 0), dtype=np.int16)

    def read_map(self, path):
        with open(path, 'r') as file:
            size = int(file.readline().strip())
            map_ = np.zeros((size, size, size), dtype=np.int16)
            for level in range(size):
                for row in range(size):
                    line = file.readline()
                    chars = np.array(list(line.strip()))
                    mask = chars == '#'
                    map_[level, row, mask] = 1 
        return map_

"""                  
class Ball:
    def __init__(self, pos=(0, 1, 1)):
        self.pos = glm.vec3(pos)
        self.up = glm.vec3(0, 1, 0)
        self.right = glm.vec3(1, 0, 0)
        self.forward = glm.vec3(0, 0, -1)
        self.pitch = 0
        self.yaw = -90
                 
    def rotate_left(self):
        self.yaw += 90

    def rotate_right(self):
        self.yaw -= 90

    def rotate_up(self):
        self.pitch += 90

    def rotate_down(self):
        self.pitch -= 90

    def move(self):
        
"""            

if __name__ == '__main__':
    a = World('maps/map1.txt')
    print(a.level_map)

    

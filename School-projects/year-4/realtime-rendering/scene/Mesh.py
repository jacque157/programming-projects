import numpy as np
from OpenGL.GL import *
import pywavefront


class BaseMesh:
    def __init__(self, app):
        self.app = app
        self.data = self.get_data()
        self.vertex_count = 0
        self.init_vao()
        self.init_vbo()
        self.program = self.get_program()

    def get_data(self):
        return None
    
    def init_vao(self):
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
    
    def init_vbo(self):
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.data.nbytes, self.data, GL_STATIC_DRAW)
        self._init_atrib_array()
    
    def _init_atrib_array(self):
        pass

    def get_program(self):
        return None
    
    def destroy(self):
        glDeleteVertexArrays(1, (self.vao, ))
        glDeleteBuffers(1, (self.vbo, ))
    
class Cube(BaseMesh):
    def __init__(self, app):
        super().__init__(app)
        self.vertex_count = len(self.data) // (3 + 3 + 2)

    def scale_to_interval(self, vertices, a, b): 
        min_ = np.min(vertices)
        max_ = np.max(vertices)
        vertices_normalised = (vertices - min_) / (max_ - min_)
        vertices_normalised *= (b - a)
        vertices_normalised += a
        return vertices_normalised
    
    def center(self, vertices):
        centre = np.mean(vertices, axis=0)
        vertices_centered = vertices - centre
        return vertices_centered

    def _init_atrib_array(self):
        vertex_size = 3
        normal_size = 3
        uv_size = 2
        bytes_ = 4
        uv_start = 0 
        normal_start = uv_size * bytes_ 
        vertex_start = (uv_size + normal_size) * bytes_

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, (vertex_size + normal_size + uv_size) * bytes_, ctypes.c_void_p(vertex_start))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, (vertex_size + normal_size + uv_size) * bytes_, ctypes.c_void_p(normal_start))
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, (vertex_size + normal_size + uv_size) * bytes_, ctypes.c_void_p(uv_start))


    def get_path(self):
        return "objects/cube/cube.obj"

    def get_data(self):
        objs = pywavefront.Wavefront(self.get_path(), parse=True)
        obj = objs.materials.popitem()[1]
        data = np.array(obj.vertices, dtype=np.float32)
        return data.flatten()
    
    def get_program(self):
        return self.app.programs['default']
    
class Skybox(Cube):
    def get_program(self):
        return self.app.programs['skybox']
    
class BeachBall(Cube):
    def __init__(self, app):
        super().__init__(app)
        self.vertex_count = len(self.data) // (3 + 3 + 2)

    def get_path(self):
        return "objects/Beach_Ball_v2/13517_Beach_Ball_v2_L3.obj"

    def get_data(self):
        objs = pywavefront.Wavefront(self.get_path(), parse=True)
        obj = objs.materials.popitem()[1]
        data = np.array(obj.vertices, dtype=np.float32)
        entry_size = 2 + 3 + 3
        x = data[5::entry_size]
        y = data[6::entry_size]
        z = data[7::entry_size]
        vertices = np.stack((x, y, z), axis=1)
        vertices = self.center(vertices)
        vertices = self.scale_to_interval(vertices, -1, 1)
        u = data[0::entry_size][:,None]
        v = data[1::entry_size][:,None]
        n_x = data[2::entry_size][:,None]
        n_y = data[3::entry_size][:,None]
        n_z = data[4::entry_size][:,None]
        data = np.concatenate((u, v, n_x, n_y, n_z, vertices), axis=1)
        return data.flatten()

    def scale_to_interval(self, vertices, a, b): 
        min_ = np.min(vertices)
        max_ = np.max(vertices)
        vertices_normalised = (vertices - min_) / (max_ - min_)
        vertices_normalised *= (b - a)
        vertices_normalised += a
        return vertices_normalised
    
    def center(self, vertices):
        centre = np.mean(vertices, axis=0)
        vertices_centered = vertices - centre
        return vertices_centered

    def get_program(self):
        return self.app.programs['default']

class Coin(BeachBall):
    def __init__(self, app):
        super().__init__(app)

    def get_path(self):
        return "objects/coin/GoldCoinBlank.obj"
    
class Apple(BeachBall):
    def __init__(self, app):
        super().__init__(app)

    def get_path(self):
        return "objects/apple/Apple.obj"
    

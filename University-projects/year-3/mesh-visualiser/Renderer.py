import tkinter as tk
import math
import time
from Structs import *

class Renderer:
 
    def __init__(self, root, w=800, h=800):
        self.width = w
        self.height = h
        self.colour = Vec4(10, 100, 100, 1)
        self.light_source = Vec4(1, 1, 2, 1)
        self.k_a = 0.5
        self.k_s = 0.5
        self.k_d = 0.5
        self.shininess = 16
        self.blin_phong = True
        self.wireframe = False
        self.point_light = True
        self.light_direction = Vec4(-1, -2, 1, 1)

        self.reset_affine_transformations()
        
        self.canvas = tk.Canvas(root, width=w, height=h)
        self.canvas.pack(side = tk.LEFT)
        
        # http://learnwebgl.brown37.net/08_projections/projections_ortho.html

        # box (2x2x2) causes clipping, using box (4x4x4) instead 
        
        left, right = -2, 2
        top, bottom = 2, -2
        near, far = -2, 2

        self.view_pos = Vec4(0, 0, far, 1)
        

        self.ortographic_projection = Renderer.create_ortographic_matrix(left, right, top, bottom, near, far)
        self.viewport_matrix = Renderer.create_viewport_matrix(w, h)
               
    def create_ortographic_matrix(left, right, top, bottom, near, far):
        mid_x = (left + right) / 2
        mid_y = (bottom + top) / 2
        mid_z = (-near + -far) / 2

        center_at_origin = Mat4(((1, 0, 0, -mid_x),
                               (0, 1, 0, -mid_y),
                               (0, 0, 1, -mid_z),
                               (0, 0, 0, 1)))
    
        scale_x = 2.0 / (right - left)
        scale_y = 2.0 / (top - bottom)
        scale_z = 2.0 / (far - near)

        scale_volume = Mat4(((scale_x, 0, 0, 0),
                            (0, scale_y, 0, 0),
                            (0, 0, scale_z, 0),
                            (0, 0, 0, 1)))

        convert_to_left_handed = Mat4(((1, 0, 0, 0),
                                        (0, 1, 0, 0),
                                        (0, 0, -1, 0),
                                        (0, 0, 0, 1)))
        
        return convert_to_left_handed * scale_volume * center_at_origin

    def create_viewport_matrix(width, height):
        offset_corner = Mat4(((1, 0, 0, width / 2),
                                (0, 1, 0, height / 2),
                                (0, 0, 1, 0),
                                (0, 0, 0, 1)))

        # scales and flips model
        scale_to_window = Mat4(((width / 2, 0, 0, 0),  
                                (0, -height / 2, 0, 0),
                                (0, 0, 1, 0),
                                (0, 0, 0, 1)))

        return offset_corner * scale_to_window
    
    def render(self, mesh):
        affine_transformations = (self.translation_matrix * self.scaling_matrix *
                                  self.rotation_x_matrix * self.rotation_y_matrix *
                                  self.rotation_z_matrix)
        
        for i, j, k in mesh.indices:
            v1 = affine_transformations * mesh.vertices[i]
            v2 = affine_transformations * mesh.vertices[j]
            v3 = affine_transformations * mesh.vertices[k]
         
            polygon_normal = Renderer.calculate_normal(v1, v2, v3).normalise()
            p1 = self.viewport_matrix * self.ortographic_projection * v1
            p2 = self.viewport_matrix * self.ortographic_projection * v2
            p3 = self.viewport_matrix * self.ortographic_projection * v3
            x_0, y_0, z_0, w = p1.components()
            x_1, y_1, z_1, w = p2.components()
            x_2, y_2, z_2, w = p3.components()

            if self.wireframe:
                x_0, y_0, z_0, w = p1.components()
                for v in (p2, p3, p1):
                    x_1, y_1, z_1, w = v.components()
                    self.canvas.create_line(x_0, y_0, x_1, y_1)
                    x_0, y_0, z_0 = x_1, y_1, z_1
            else:
                if (self.view_pos.normalise().dot_product(polygon_normal) > 0):
                    centre = Renderer.calculate_centre(v1, v2, v3)
                    light_direction = (self.light_source - centre).normalise() if self.point_light \
                                      else self.light_direction.normalise()
                    view_direction = (self.view_pos - centre).normalise()
                    
                    intensity = Renderer.calculate_intensity(polygon_normal, view_direction,
                                                             light_direction, self.k_a,
                                                             self.k_d, self.k_s, self.shininess,
                                                             self.blin_phong)
                    
                    r, g, b, w = map(lambda a: min(a, 255), map(
                        int, (intensity  * self.colour).components()))

                    self.canvas.create_polygon(x_0, y_0, x_1, y_1, x_2, y_2, fill=f"#{r:02x}{g:02x}{b:02x}")

    def calculate_intensity(n, v, l, k_a, k_d, k_s, s, blin_phong):
        i_d = max(n.dot_product(l), 0)  
        if blin_phong:
            h = (v + l).normalise()
            i_s = max(n.dot_product(h) ** s, 0) 
        else:
            # https://math.stackexchange.com/questions/13261/how-to-get-a-reflection-vector
            r = (l - (2 * (l.dot_product(n)) * n)).normalise()
            i_s = max(r.dot_product(v) ** s, 0) 
        i_a = i_s + i_d
        i = ((i_a * k_a) + (i_d * k_d) + (i_s * k_s))
        return i
                
    def calculate_normal(p1, p2, p3):
        v1 = p2 - p1
        v2 = p3 - p2
        return v1 * v2

    def calculate_centre(p1, p2, p3):
        x1, y1, z1, w1 = p1.components()
        x2, y2, z2, w2 = p2.components()
        x3, y3, z3, w3 = p3.components()
        return Vec4((x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3, (z1 + z2 + z3) / 3, (w1 + w2 + w3) / 3)

    def clear(self):
        self.canvas.delete("all")

    def update(self):
        self.canvas.update()

    def set_translation_matrix(self, dx, dy, dz):
        self.translation_matrix = Mat4(((1, 0, 0, dx),
                                       (0, 1, 0, dy),
                                       (0, 0, 1, dz),
                                       (0, 0, 0, 1)))

    def set_scaling_matrix(self, sx, sy, sz):
        self.scaling_matrix = Mat4(((sx, 0, 0, 0),
                                       (0, sy, 0, 0),
                                       (0, 0, sz, 0),
                                       (0, 0, 0, 1)))

    def set_rotation_x_matrix(self, deg):
        cos = math.cos(math.radians(deg))
        sin = math.sin(math.radians(deg))
        self.rotation_x_matrix = Mat4(((1, 0, 0, 0),
                                       (0, cos, -sin, 0),
                                       (0, sin, cos, 0),
                                       (0, 0, 0, 1)))

    def set_rotation_y_matrix(self, deg):
        cos = math.cos(math.radians(deg))
        sin = math.sin(math.radians(deg))
        self.rotation_y_matrix = Mat4(((cos, 0, sin, 0),
                                       (0, 1, 0, 0),
                                       (-sin, 0, cos, 0),
                                       (0, 0, 0, 1)))
        
    def set_rotation_z_matrix(self, deg):
        cos = math.cos(math.radians(deg))
        sin = math.sin(math.radians(deg))
        self.rotation_z_matrix = Mat4(((cos, -sin, 0, 0),
                                       (sin, cos, 0, 0),
                                       (0, 0, 1, 0),
                                       (0, 0, 0, 1)))

    def set_light_position(self, x, y, z):
        self.light_source = Vec4(x, y, z, 1)

    def get_light_position(self):
        return self.light_source.components()[0:-1]

    def set_light_direction(self, dx, dy, dz):
        self.light_direction = -1 * Vec4(dx, dy, dz, 1)

    def get_light_direction(self):
        return (-1 * self.light_direction).components()[0:-1]

    def set_colour(self, r, g, b):
        self.colour = Vec4(r, g, b, 1)

    def get_colour(self):
        return self.colour.components()[0:-1]

    def set_ambient_reflection(self, a):
        self.k_a = a

    def get_ambient_reflection(self):
        return self.k_a

    def set_diffuse_reflection(self, d):
        self.k_d = d

    def get_diffuse_reflection(self):
        return self.k_d

    def set_specular_reflection(self, s):
        self.k_s = s

    def get_specular_reflection(self):
        return self.k_s

    def set_shininess(self, h):
        self.shininess = h

    def get_shininess(self):
        return self.shininess
        
    def reset_affine_transformations(self):
        self.set_translation_matrix(0, 0, 0)
        self.set_scaling_matrix(1, 1, 1)
        self.set_rotation_x_matrix(0)
        self.set_rotation_y_matrix(0)
        self.set_rotation_z_matrix(0)
        
        
    def play_demo(self):
        obj_1 = Mesh("obj_files/bunny.obj") 
        obj_2 = Mesh("obj_files/icosphere.obj")
        obj_3 = Mesh("obj_files/monkey.obj")
  
        while True:
            for obj in obj_1, obj_2, obj_3:
                self.clear()
                self.render(obj)
                self.canvas.update()
                time.sleep(3)

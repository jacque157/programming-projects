from OpenGL.GL import *
import glm
import numpy as np
import ctypes
import pygame as pg


DEFAULT = 0
OREN_NAYAR = 1
COOK_TORRANCE = 2


class BaseModel:
    def __init__(self, app, name, texture_name, program_name='default', 
                 position=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1), 
                 Ia=0.06, Id=0.8, Is=1.0, shine=32.0, 
                 roughness=0.0, fresnel=0.0, shading=DEFAULT, casts_shadow=False):
        self.app = app
        self.name = name
        self.program_name = program_name
        self.texture = self.app.textures[texture_name]
        self.shadow_map = self.app.textures['shadow_map']
        self.pos = glm.vec3(position)
        self.rot = glm.vec3(rotation)
        self.scale = glm.vec3(scale)
        self.Ia = Ia 
        self.Id = Id
        self.Is = Is# 1.0 * self.colour
        self.shine = shine
        self.roughness = roughness
        self.F0 = fresnel
        self.shading = shading

        self.mesh = app.meshes[name]
        self.camera = app.camera
        self.shader = app.programs[program_name]
        self.shadow_shader = app.programs['shadow_map']
        self.casts_shadow = casts_shadow

        self.m_model = self.get_model_matrix()
        self.init_mvp()
        self.init_material()
        self.init_main_light()
        self.init_lights()
        self.init_resolution()
        self.init_shadow_map_shader()

    def get_model_matrix(self):
        m_model = glm.translate(glm.mat4(), self.pos)
        rot = glm.radians(self.rot)
        m_model = glm.rotate(m_model, rot.x, glm.vec3(1, 0, 0))
        m_model = glm.rotate(m_model, rot.y, glm.vec3(0, 1, 0))
        m_model = glm.rotate(m_model, rot.z, glm.vec3(0, 0, 1))
        m_model = glm.scale(m_model, self.scale)
        return m_model
    
    def render(self):
        pass

    def init_lights(self):
        glUseProgram(self.shader.program)
        lights = self.app.scene.lights 
        self.shader.set_int('pointLightsCount', ctypes.c_int(len(lights)))
        self.shader.set_int('shading', ctypes.c_int(self.shading))
        self.shader.set_bool('toonShading', self.app.toon_shading)

        for i, light in enumerate(lights):
            self.shader.set_vector_3f(f'pointLights[{i}].position', np.array(light.pos1, dtype=np.float32))
            self.shader.set_vector_3f(f'pointLights[{i}].colour', np.array(light.colour, dtype=np.float32))
            self.shader.set_vector_3f(f'pointLights[{i}].attenuation', np.array(light.attenuation, dtype=np.float32))
        

    def init_shadow_map_shader(self):
        glUseProgram(self.shadow_shader.program)
        main_light = self.app.scene.main_light
        self.shadow_shader.set_matrix_f4x4('lightSpaceMatrix', np.array(main_light.light_space_matrix, dtype=np.float32))
        self.shadow_shader.set_matrix_f4x4('m_model', np.array(self.m_model, dtype=np.float32))

    def init_mvp(self):
        glUseProgram(self.shader.program)
        self.shader.set_matrix_f4x4('m_proj', np.array(self.camera.m_proj, dtype=np.float32))
        self.shader.set_matrix_f4x4('m_view', np.array(self.camera.m_view, dtype=np.float32))
        self.shader.set_matrix_f4x4('m_model', np.array(self.m_model, dtype=np.float32))

    def init_main_light(self):
        glUseProgram(self.shader.program)
        main_light = self.app.scene.main_light 
        self.shader.set_vector_3f('mainLight.position', np.array(main_light.pos, dtype=np.float32))
        self.shader.set_vector_3f('mainLight.direction', np.array(main_light.dir, dtype=np.float32))
        self.shader.set_vector_3f('mainLight.colour', np.array(main_light.colour, dtype=np.float32))
        self.shader.set_matrix_f4x4('lightSpaceMatrix', np.array(main_light.light_space_matrix, dtype=np.float32))
        
    def init_material(self):
        glUseProgram(self.shader.program)
        self.shader.set_float('material.Ia', ctypes.c_float(self.Ia))
        self.shader.set_float('material.Id', ctypes.c_float(self.Id))
        self.shader.set_float('material.Is', ctypes.c_float(self.Is))
        self.shader.set_float('material.shine', ctypes.c_float(self.shine))
        self.shader.set_float('material.roughness', ctypes.c_float(self.roughness))
        self.shader.set_float('material.F0', ctypes.c_float(self.F0))
        
    def init_resolution(self):
        glUseProgram(self.shader.program)
        self.shader.set_vector_2f('u_resolution', np.array((self.shadow_map.width, self.shadow_map.height), dtype=np.float32))

    def update(self):
        glUseProgram(self.shader.program)
        self.shader.set_matrix_f4x4('m_model', np.array(self.m_model, dtype=np.float32))
        self.shader.set_matrix_f4x4('m_view', np.array(self.camera.m_view, dtype=np.float32))
        main_light = self.app.scene.main_light
        self.shader.set_matrix_f4x4('lightSpaceMatrix', np.array(main_light.light_space_matrix, dtype=np.float32))
        self.shader.set_bool('toonShading', self.app.toon_shading)
        self.shader.set_int('shading', ctypes.c_int(self.shading))
        self.shader.set_float('material.Ia', ctypes.c_float(self.Ia))
        self.shader.set_float('material.Id', ctypes.c_float(self.Id))
        self.shader.set_float('material.Is', ctypes.c_float(self.Is))
        self.shader.set_float('material.shine', ctypes.c_float(self.shine))
        self.shader.set_float('material.roughness', ctypes.c_float(self.roughness))
        self.shader.set_float('material.F0', ctypes.c_float(self.F0))

    def update_shadow(self):
        glUseProgram(self.shadow_shader.program)
        self.shadow_shader.set_matrix_f4x4('m_model', np.array(self.m_model, dtype=np.float32))

class CubeModel(BaseModel):
    def __init__(self, app, name='cube', texture_name='egyptian_wall', program_name='default', 
                 position=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1), 
                 Ia=0.06, Id=0.8, Is=1.0, shine=32.0, 
                 roughness=0.0, fresnel=0.0, shading=DEFAULT, casts_shadow=True):
        super().__init__(app, name, texture_name, program_name, position, rotation, scale, Ia, Id, Is, shine, roughness, fresnel, shading, casts_shadow)

    def render(self):
        glUseProgram(self.shader.program)
        self.texture.use()
        self.shadow_map.use()
        glBindVertexArray(self.mesh.vao)
        glDrawArrays(GL_TRIANGLES, 0, self.mesh.vertex_count)

    def render_shadow(self):
        glUseProgram(self.shadow_shader.program)
        #self.texture.use()
        #self.shadow_map.use()
        glBindVertexArray(self.mesh.vao)
        glDrawArrays(GL_TRIANGLES, 0, self.mesh.vertex_count)


class StoneWall(CubeModel):
    def __init__(self, app, name='cube', texture_name='stone_wall', program_name='default', 
                 position=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1),
                 Ia=0.06, Id=0.8, Is=1.0, shine=32.0, 
                 roughness=20.0, fresnel=0.0, shading=COOK_TORRANCE, casts_shadow=True):
        super().__init__(app, name, texture_name, program_name, position, rotation, scale, Ia, Id, Is, shine, roughness, fresnel, shading, casts_shadow)

class Tiles(CubeModel):
    def __init__(self, app, name='cube', texture_name='gray', program_name='default', 
                 position=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1),
                 Ia=0.06, Id=0.8, Is=1.0, shine=32.0, 
                 roughness=0.0, fresnel=0.0, shading=DEFAULT, casts_shadow=True):
        super().__init__(app, name, texture_name, program_name, position, rotation, scale, Ia, Id, Is, shine, roughness, fresnel, shading, casts_shadow)

class BeachBall(CubeModel):
    def __init__(self, app, name='beach_ball', texture_name='beach_ball', program_name='default', 
                 position=(0, 0, 0), rotation=(90, 0, 0), scale=(1.0, 1.0, 1.0), 
                 Ia=0.06, Id=0.8, Is=1.0, shine=32.0, 
                 roughness=0.0, fresnel=0.0, shading=DEFAULT, casts_shadow=True):
        super().__init__(app, name, texture_name, program_name, position, rotation, scale, Ia, Id, Is, shine, roughness, fresnel, shading, casts_shadow)

class Coin(BeachBall):
    def __init__(self, app, name='coin', texture_name='coin', program_name='default', 
                 position=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1), 
                 Ia=0.06, Id=0.8, Is=1, shine=32,
                 roughness=0.5, fresnel=1.0, shading=2, casts_shadow=True):
        super().__init__(app, name, texture_name, program_name, position, rotation, scale, Ia, Id, Is, shine, roughness, fresnel, shading, casts_shadow)

class Apple(BeachBall):
    def __init__(self, app, name='apple', texture_name='apple', program_name='default', 
                 position=(0, 0, 0), rotation=(0, 0, 0), scale=(0.5, 0.5, 0.5), 
                 Ia=0.06, Id=0.8, Is=1, shine=32,
                 roughness=0.0, fresnel=0.0, shading=0, casts_shadow=True):
        super().__init__(app, name, texture_name, program_name, position, rotation, scale, Ia, Id, Is, shine, roughness, fresnel, shading, casts_shadow)

class Cat(CubeModel):
    def __init__(self, app, name='cat', texture_name='cat', program_name='default', 
                 position=(0, 0, 0), rotation=(270, 0, 0), scale=(0.5, 0.5, 0.5), 
                 Ia=0.06, Id=0.8, Is=1, shine=32,
                 roughness=0.5, fresnel=0.0, shading=1, casts_shadow=True):
        super().__init__(app, name, texture_name, program_name, position, rotation, scale, Ia, Id, Is, shine, roughness, fresnel, shading, casts_shadow)

class SkyBoxModel(BaseModel):
    def __init__(self, app, name='skybox', texture_name='skybox', program_name='skybox', position=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1), casts_shadow=False):
        super().__init__(app, name, texture_name, program_name, position, rotation, scale, casts_shadow=casts_shadow)

    def get_model_matrix(self):
        return glm.mat4(1.0)
    
    def render(self):
        glUseProgram(self.shader.program)
        self.texture.use()
        glBindVertexArray(self.mesh.vao)
        glDrawArrays(GL_TRIANGLES, 0, self.mesh.vertex_count)

    def init_shadow_map_shader(self):
        pass

    def init_mvp(self):
        glUseProgram(self.shader.program)
        self.shader.set_matrix_f4x4('m_proj', np.array(self.camera.m_proj, dtype=np.float32))
        view = glm.mat4(glm.mat3(self.camera.m_view))
        self.shader.set_matrix_f4x4('m_view', np.array(view, dtype=np.float32))
        self.shader.set_matrix_f4x4('m_model', np.array(self.m_model, dtype=np.float32))

    def init_main_light(self):
        pass

    def init_resolution(self):
        pass

    def update(self):
        glUseProgram(self.shader.program)
        view = glm.mat4(glm.mat3(self.camera.m_view))
        self.shader.set_matrix_f4x4('m_view', np.array(view, dtype=np.float32))
        #self.shader.set_bool('toonShading', self.app.toon_shading)
        
    def update_shadow(self):
        pass

    def init_material(self):
        pass

    def init_lights(self):
        pass


class Player(BeachBall):
    def __init__(self, app, x, y, z, map_, objects, x_offset, z_offset):
        self.vertical_axis = glm.vec3(0, 1, 0)
        self.vertical_axis_angle = 90
        self.horizontal_axis_angle = 0
        self.dva = 0.1
        self.speed = 0.005
        self.dha = self.speed * 180 / np.pi
        self.map_ = map_
        self.objects = objects
        self.x_offset = x_offset
        self.z_offset = z_offset
        super().__init__(app, name='beach_ball', texture_name='beach_ball', program_name='default', 
                         position=(x, y, z), rotation=(0, 0, 0), scale=(0.5, 0.5, 0.5), 
                         Ia=0.06, Id=0.8, Is=1, 
                         shine=32, roughness=0, fresnel=0, 
                         shading=DEFAULT, casts_shadow=True)

    def get_colliding(self):
        z, x = int(((self.pos.z + self.z_offset) / 2) + 0.5), int(((self.pos.x + self.x_offset) / 2) + 0.5)
        return self.objects.get((x, z), None)
    
    def is_colliding(self):
        z, x = int(np.floor(((self.pos.z + self.z_offset) / 2) + 0.5)), int(np.floor(((self.pos.x + self.x_offset) / 2) + 0.5))
        if 0 <= z < len(self.map_) and 0 <= x < len(self.map_[z]) and self.map_[z][x] == '#':
            return False
        return True

    def get_forward_vector(self):
        z = np.cos(np.deg2rad(self.vertical_axis_angle))
        x = np.sin(np.deg2rad(self.vertical_axis_angle))
        return glm.normalize(glm.vec3(x, 0, z))

    def rotate_along_vertical_axis(self, da):
        self.vertical_axis_angle += da
        if self.vertical_axis_angle > 360:
            self.vertical_axis_angle -= 360
        elif self.vertical_axis_angle < 0:
            self.vertical_axis_angle += 360

    def rotate_along_horizontal_axis(self, da):     
        self.horizontal_axis_angle += da
        if self.horizontal_axis_angle > 360:
            self.horizontal_axis_angle -= 360
        elif self.horizontal_axis_angle < 0:
            self.horizontal_axis_angle += 360

    def move(self, speed):
        self.pos += self.get_forward_vector() * speed

    def get_model_matrix(self):
        m_model = glm.translate(glm.mat4(), self.pos)
        rotational_axis = glm.cross(self.vertical_axis, self.get_forward_vector())
        m_model = glm.rotate(m_model, np.deg2rad(self.horizontal_axis_angle), -rotational_axis)
        rot = glm.radians(self.rot)
        #m_model = glm.rotate(m_model, rot.x, glm.vec3(1, 0, 0))
        m_model = glm.rotate(m_model, np.deg2rad(self.vertical_axis_angle), glm.vec3(0, 1, 0))
        #m_model = glm.rotate(m_model, rot.z, glm.vec3(0, 0, 1))

        
        m_model = glm.scale(m_model, self.scale)
        return m_model

    def update(self):
        ds = self.speed * self.app.delta_time
        dav = self.dva * self.app.delta_time
        dah = self.dha * self.app.delta_time

        prev_angle = self.horizontal_axis_angle
        prev_pos = glm.vec3(self.pos.xyz)
        keys = pg.key.get_pressed()
        if keys[pg.K_UP]:
            self.move(ds)
            self.rotate_along_horizontal_axis(-dah)
        if keys[pg.K_LEFT]:
            self.rotate_along_vertical_axis(-dav)
        if keys[pg.K_RIGHT]:
            self.rotate_along_vertical_axis(dav)

        if self.is_colliding():
            self.pos = prev_pos
            self.horizontal_axis_angle = prev_angle
        self.m_model = self.get_model_matrix()

        object_ = self.get_colliding()
        if object_:
            object_.collected = True

        super().update()

class AnimatedCoin(Coin):
    def __init__(self, app, x, y, z):
        self.collected = False
        super().__init__(app, name='coin', texture_name='coin', program_name='default', 
                         position=(x, y, z), rotation=(0, np.random.randint(0, 360), 0), scale=(0.5, 0.5, 0.5), 
                         Ia=0.06, Id=0.8, Is=1, 
                         shine=32, roughness=0.5, fresnel=1, shading=2, casts_shadow=True)

    def update(self):
        self.rot.y += 1
        if self.rot.y > 360:
            self.rot.y -= 360
        self.m_model = self.get_model_matrix()
        super().update()

class AnimatedApple(Apple):
    def __init__(self, app, x, y, z):
        self.collected = False
        super().__init__(app, name='apple', texture_name='apple', program_name='default', 
                         position=(x, y, z), rotation=(0, 0, 0), scale=(0.5, 0.5, 0.5), 
                         Ia=0.06, Id=0.5, Is=1, shine=32, 
                         roughness=0, fresnel=0, shading=0, casts_shadow=True)
        
    def update(self):
        self.rot.y += 1
        self.rot.x += 1
        self.rot.z += 1
        if self.rot.y > 360:
            self.rot.y -= 360
        if self.rot.x > 360:
            self.rot.x -= 360
        if self.rot.z > 360:
            self.rot.z -= 360
        self.m_model = self.get_model_matrix()
        super().update()
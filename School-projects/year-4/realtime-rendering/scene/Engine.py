import pygame as pg
from OpenGL.GL import *
import numpy as np
import sys

from Camera import Camera
import Shader
import Mesh
from Scene import *
import Texture

class Engine:
    def __init__(self, width=1024, height=640, scene=Scene2) -> None:
        self.width= width
        self.height = height
        self.time = 0
        self.delta_time = 0
        self.toon_shading = False

        self.camera = Camera(self)

        self.init_pg()
        self.init_gl()
        self.clock = pg.time.Clock()
        self.programs = {}
        self.init_programs()
        self.meshes = {}
        self.init_meshes()
        self.textures = {}
        self.init_textures()
        
        self.scene = scene(self)
        self.scene.init_scene()

    def init_pg(self):
        pg.init()
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
        pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, 
                                    pg.GL_CONTEXT_PROFILE_CORE)
        pg.display.set_mode((self.width, self.height), 
                            flags=pg.OPENGL | pg.DOUBLEBUF)
        pg.event.set_grab(True)
        pg.mouse.set_visible(False)

    def init_gl(self):

        glEnable(GL_MULTISAMPLE)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)

    def init_programs(self):
        self.programs['default'] = Shader.DefaultShader()
        self.programs['shadow_map'] = Shader.ShadowMap()
        self.programs['skybox'] = Shader.SkyBox()

    def init_meshes(self):
        self.meshes['cube'] = Mesh.Cube(self)
        self.meshes['skybox'] = Mesh.Skybox(self)
        self.meshes['beach_ball'] = Mesh.BeachBall(self)
        self.meshes['coin'] = Mesh.Coin(self)
        self.meshes['apple'] = Mesh.Apple(self)

    def init_textures(self):
        self.textures['egyptian_wall'] = Texture.Texture(self, 'textures/egypt_wall.png')
        self.textures['stone_wall'] = Texture.Texture(self, 'textures/stone_wall.jpg')
        self.textures['beach_ball'] = Texture.Texture(self, 'objects/Beach_Ball_v2/Beach_Ball_diffuse.jpg')
        self.textures['gray'] = Texture.Texture(self, 'textures/gray.png')
        self.textures['gold'] = Texture.Texture(self, 'textures/gold.png')
        self.textures['red'] = Texture.Texture(self, 'textures/red.png')
        self.textures['wood'] = Texture.Texture(self, 'textures/wood.png')
        self.textures['metal1'] = Texture.Texture(self, 'textures/metal.png')
        self.textures['metal2'] = Texture.Texture(self, 'textures/metal2.png')
        self.textures['tiles'] = Texture.Texture(self, 'textures/tiles.jpg')
        self.textures['coin'] = Texture.Texture(self, 'objects/coin/GoldCoinBlank.png')
        self.textures['apple'] = Texture.Texture(self, 'objects/apple/Apple.png')
        self.textures['shadow_map'] = Texture.ShadowMap(self)
        self.textures['skybox'] = Texture.Skybox(self)

    def handle_events(self):      
        for event in pg.event.get():
            if event.type == pg.QUIT or \
                (event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE):
                self.destroy()
                pg.quit()
                sys.exit()
            if event.type == pg.KEYDOWN and event.key == pg.K_t:
                self.toon_shading = not self.toon_shading
            if event.type == pg.KEYDOWN and event.key == pg.K_l:       
                self.scene.main_light.colour -=  glm.vec3(0.1, 0.1, 0.1)
                self.scene.main_light.colour = glm.clamp(self.scene.main_light.colour, 0.0, 1.0)
            if event.type == pg.KEYDOWN and event.key == pg.K_p:
                self.scene.main_light.colour +=  glm.vec3(0.1, 0.1, 0.1)
                self.scene.main_light.colour = glm.clamp(self.scene.main_light.colour, 0.0, 1.0)

    def render(self):
        self.scene.render()    
        pg.display.flip()

    def destroy(self):
        for program in self.programs.values():
            program.destroy()
        
        for mesh in self.meshes.values():
            mesh.destroy()

        for texture in self.textures.values():
            texture.destroy()
        
    def run(self):
        while True:
            self.set_time()
            self.handle_events()
            self.camera.update()
            self.render()
            self.delta_time = self.clock.tick(60)

    def set_time(self):
        self.time = pg.time.get_ticks() * 0.001

if __name__ == '__main__':
    app = Engine()
    app.run()
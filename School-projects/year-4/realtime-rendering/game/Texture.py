import pygame as pg
from OpenGL.GL import *
import numpy as np


class Texture:
    def __init__(self, app, path):
        self.app = app
        self.texture_id = 0
        self.texture_uniform = None
        self.init_texture()
        self.load_image(path)
        
    def init_texture(self):
        self.texture_uniform = self.app.programs['default'].get_uniform_id('u_texture_0')
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    def load_image(self, path):
        image = pg.image.load(path).convert()
        image = pg.transform.flip(image, 
                                  flip_x=False, 
                                  flip_y=True)
        width, height = image.get_rect().size
        image_data = pg.image.tostring(image, 'RGB')
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image_data)
        glGenerateMipmap(GL_TEXTURE_2D)

    def use(self):
        glUseProgram(self.app.programs['default'].program)
        glUniform1i(self.texture_uniform, 0) # send to unit 0
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)

    def destroy(self):
        glDeleteTextures(1, (self.texture_id, ))

class Skybox:
    def __init__(self, app, directory='skybox', ext='png'):
        self.app = app
        self.texture_id = 0
        self.texture_uniform = None
        self.init_texture()
        self.load_images(directory, ext)

    def init_texture(self):
        self.texture_uniform = self.app.programs['default'].get_uniform_id('u_texture_0')
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_CUBE_MAP, self.texture_id)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)

    def load_images(self, directory, ext):
        faces = ('right', 'left', 'top', 'bottom', 'back', 'front')
        images = []
        for face in faces:
            image = pg.image.load(f'{directory}/{face}.{ext}').convert()
            if face in ['right', 'left', 'front', 'back']:
                image = pg.transform.flip(image, flip_x=True, flip_y=False)
            else:
                image = pg.transform.flip(image, flip_x=False, flip_y=True)
            images.append(image)

        width, height = images[0].get_size()

        for i in range(6):
            texture = pg.image.tostring(images[i], 'RGB')
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                        0, GL_RGB, width, height, 0, 
                        GL_RGB, GL_UNSIGNED_BYTE, texture)
    def use(self):
        glUseProgram(self.app.programs['skybox'].program)
        #glUniform1i(self.texture_uniform, 0) # send to unit 0
        #glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_CUBE_MAP, self.texture_id)

    def destroy(self):
        glDeleteTextures(1, (self.texture_id, ))
   
class ShadowMap:
    def __init__(self, app):
        self.app = app
        self.fbo = 0
        self.texture_id = 0
        self.width = 1024# app.width #* 4
        self.height = 1024# app.height #* 4
        self.init_texture()

    def init_texture(self):
        self.texture_uniform = self.app.programs['default'].get_uniform_id('shadowMap')
        self.fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, self.width,
                     self.height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, 
                          np.ones(4, dtype=np.float32))
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D,
                               self.texture_id, 0)
        glDrawBuffer(GL_NONE)
        glReadBuffer(GL_NONE)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
    
    def use(self):
        glUseProgram(self.app.programs['default'].program)
        glUniform1i(self.texture_uniform, 1) # send to unit 1
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)

    def destroy(self):
        glDeleteTextures(1, (self.texture_id,))
        glDeleteFramebuffers(1, (self.fbo,))

import glm
import pygame as pg 

FOV = 50
NEAR = 0.1
FAR = 100
SPEED = 0.01
SENSITIVITY = 0.05


class Camera:
    def __init__(self, app, position=(0, 0, 4), yaw=-90, pitch=0):
        self.app = app
        self.aspect_ratio = app.width / app.height
        self.pos = glm.vec3(position)
        self.yaw = yaw
        self.pitch = pitch
        self.up = glm.vec3(0, 1, 0)
        self.right = glm.vec3(1, 0, 0)
        self.forward = glm.vec3(0, 0, -1)
        self.m_view = self.get_view()
        self.m_proj = self.get_projection()

    def rotate(self):
        rel_x, rel_y = pg.mouse.get_rel()
        self.yaw += rel_x * SENSITIVITY
        self.pitch -= rel_y * SENSITIVITY
        self.pitch = glm.clamp(self.pitch, -89, 89)
        
    def update_camera_vectors(self):
        yaw, pitch = glm.radians(self.yaw), glm.radians(self.pitch)
        self.forward.x = glm.cos(yaw) * glm.cos(pitch)
        self.forward.y = glm.sin(pitch)
        self.forward.z = glm.sin(yaw) * glm.cos(pitch)
        self.forward = glm.normalize(self.forward)
        self.right = glm.normalize(glm.cross(self.forward, glm.vec3(0, 1, 0)))
        self.up = glm.normalize(glm.cross(self.right, self.forward))

    def update(self):
        self.move()
        self.rotate()
        self.update_camera_vectors()
        self.m_view = self.get_view()

    def move(self):
        velocity = SPEED * self.app.delta_time
        keys = pg.key.get_pressed()
        if keys[pg.K_w]:
            self.pos += self.forward * velocity
        if keys[pg.K_s]:
            self.pos -= self.forward * velocity
        if keys[pg.K_a]:
            self.pos -= self.right * velocity
        if keys[pg.K_d]:
            self.pos += self.right * velocity
        if keys[pg.K_q]:
            self.pos += self.up * velocity
        if keys[pg.K_e]:
            self.pos -= self.up * velocity

    def get_projection(self):
        return glm.perspective(glm.radians(FOV), self.aspect_ratio, NEAR, FAR)
    
    def get_view(self):
        return  glm.lookAt(self.pos, 
                           self.pos + self.forward, 
                           self.up)
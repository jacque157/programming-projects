import glm


"""class Light:
    def __init__(self, position=(25, 25, -5), colour=(1, 1, 1)):
        self.is_point = True
        self.a = glm.vec3(position)
        self.colour = glm.vec3(colour)
        self.b = glm.vec3(0, 0, 0)

        self.Ia = 0.06 * self.colour
        self.Id = 0.8 * self.colour
        self.Is = 0.0 * self.colour# 1.0 * self.colour

        self.light_space_matrix = self.get_light_space_matrix()

    def get_light_space_matrix(self):
        near_plane = 1.0
        far_plane = 100.0
        projection = glm.ortho(-10.0, 10.0, -10.0, 10.0, near_plane, far_plane); 
        if self.is_point:
            view = glm.lookAt(self.a, self.b, glm.vec3(0, 1, 0)) 
        else:
            view = glm.lookAt(self.b - self.a, self.b, glm.vec3(0, 1, 0))
        return projection * view"""

class DirectionalLight:
    def __init__(self, app, position=(25.0, 25.0, -5.0), direction=(-25.0, -25.0, 5.0), colour=(1.0, 1.0, 1.0)) -> None:
        self.app = app
        self.pos = glm.vec3(position)
        self.colour = glm.vec3(colour)
        self.dir = glm.vec3(direction)
        self.light_space_matrix = self.get_light_space_matrix()

    def get_light_space_matrix(self):
        near_plane = 1.0
        far_plane = 100.0
        projection = glm.ortho(-10.0, 10.0, -10.0, 10.0, near_plane, far_plane); 
        view = glm.lookAt(self.pos, self.pos + self.dir, glm.vec3(0, 1, 0))
        return projection * view

class PointLight:
    def __init__(self, app, position1=(25.0, 25.0, -5.0), position2=(0.0, -0.0, 0.0), colour=(1.0, 1.0, 1.0), attenuation=(1.0, 0.0, 0.0)) -> None:
        self.app = app
        self.pos1 = glm.vec3(position1)
        self.pos2 = glm.vec3(position2)
        self.colour = glm.vec3(colour)
        self.attenuation = glm.vec3(attenuation)
        self.light_space_matrix = self.get_light_space_matrix()

    def get_light_space_matrix(self):
        projection = self.app.camera.m_proj
        view = glm.lookAt(self.pos1, self.pos2, glm.vec3(0, 1, 0)) 
        return projection * view
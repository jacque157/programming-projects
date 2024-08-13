from Model import *
from Light import *


class Scene:
    def __init__(self, app):
        self.app = app
        self.models = []
        self.main_light = None
        self.lights = []
        self.shadow_map = app.textures['shadow_map']
        self.skybox = None
        glClearColor(0.1, 0.2, 0.2, 1)
        
    def init_scene(self):
        pass

    def add_model(self, model):
        self.models.append(model)

    def render(self):
        #glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        #glBindFramebuffer(GL_FRAMEBUFFER, self.shadow_map.fbo)
        glViewport(0, 0, self.shadow_map.width, self.shadow_map.height)
        glBindFramebuffer(GL_FRAMEBUFFER, self.shadow_map.fbo)
        glClear(GL_DEPTH_BUFFER_BIT)
        glCullFace(GL_FRONT)
        for model in self.models:    
            if model.casts_shadow: 
                model.update_shadow()                 
                model.render_shadow()
        glCullFace(GL_BACK)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(0, 0, self.app.width, self.app.height)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glDepthMask(GL_FALSE)
        
        if self.skybox is not None:
            self.skybox.update()   
            self.skybox.render()
        glDepthMask(GL_TRUE)
        for model in self.models:    
            model.update()   
            model.render()

        

class Scene1(Scene):
    def __init__(self, app):
        super().__init__(app)
        
    def init_scene(self):
        self.skybox = SkyBoxModel(self.app)
        self.main_light = DirectionalLight(self.app)

        self.lights.append(PointLight(self.app, position1=(-10, 3, 1), position2=(-10, 0, 1), colour=(1, 0, 0), attenuation=(1.5, 0.25, 0)))
        self.lights.append(PointLight(self.app, position1=(0, 3, 15), position2=(0, 0, 15), colour=(0, 1, 0), attenuation=(1.5, 0.5, 0)))

        add = self.add_model
        cube1 = StoneWall(self.app, position=(0, 2, 0))
        add(cube1)
        cube2 = StoneWall(self.app, position=(1, 2, 0))
        add(cube2)

        ball = BeachBall(self.app, position=(0, 0, 0))
        add(ball)
        cube2 = StoneWall(self.app, position=(10, 1, 0))
        add(cube2)

        #cube3 = StoneWall(self.app, position=(25.0, 25.0, -5.0))
        #add(cube3)
        n, s = 20, 2

        for x in range(-n, n, s):
            for z in range(-n, n, s):
                 add(CubeModel(self.app, position=(x, -s, z)))

        """for i in range(9):
            add(StoneWall(self.app, position=(15, i * s, -9 + i)))
            add(StoneWall(self.app, position=(15, i * s, 5 - i)))"""

    
class Scene2(Scene):
    def __init__(self, app):
        super().__init__(app)

        
    def init_scene(self):
        self.skybox = SkyBoxModel(self.app)
        self.main_light = DirectionalLight(self.app, position=(0, 10, 15), direction=(0, -10, -15))

        add = self.add_model
        
        ball = BeachBall(self.app, position=(0, 0, -8), texture_name='gold', shading=0)
        add(ball)

        ball = Apple(self.app, position=(0, 0, 8))
        add(ball)

        start_x = -6
        start_z = -4
        self.lights.append(PointLight(self.app, position1=(start_x + 4, 5, start_z + 4), position2=(0, 0, 0), colour=(1, 0, 0), attenuation=(1.5, 0.5, 0)))
        for i, r in enumerate((0.0, 0.5, 1.0)):
            for j, f in enumerate((0.0, 0.5, 1.0)):
                x = start_x + 4 * i
                z = start_z + 4 * j

                ball = BeachBall(self.app, position=(x, 0, z), texture_name='gold', shading=2, roughness=r, fresnel=f)
                #self.lights.append(PointLight(self.app, position1=(-2, 5, 0), position2=(0, 0, 0), colour=(1, 1, 1), attenuation=(1.5, 0.5, 0)))
                add(ball)

        start_x += 4 * 3
        start_z = start_z
        self.lights.append(PointLight(self.app, position1=(start_x + 4, 5, start_z + 4), position2=(0, 0, 0), colour=(0, 1, 0), attenuation=(1.5, 0.5, 0)))
        for i, r in enumerate(reversed((0.0, 0.5, 1.0))):
            x = start_x + 4 * 0
            z = start_z + 4 * i
            ball = BeachBall(self.app, position=(x, 0, z), texture_name='gold', shading=1, roughness=15*r, fresnel=0.0)
            #self.lights.append(PointLight(self.app, position1=(-2, 5, 0), position2=(0, 0, 0), colour=(1, 1, 1), attenuation=(1.5, 0.5, 0)))
            add(ball)

        n, s = 20, 2

        for x in range(-n, n, s):
            for z in range(-n, n, s):
                 add(Tiles(self.app, position=(x, -s, z)))

class Game(Scene):
    def __init__(self, app):
        self.player = None
        self.map_, self.objects = self.read_file('maps/map1.txt')
        super().__init__(app)

    def read_file(self, path):
        map_ = []
        objects = {}
        with open(path) as file:
            lines = int(file.readline().strip())
            for _ in range(lines):
                line = list(file.readline().strip())
                map_.append(line)
            n = int(file.readline().strip())
            for _ in range(n):
                line = file.readline().strip().split()
                z, x, type_ = int(line[1]), int(line[2]), line[0]
                objects[(x, z)] = type_
        return map_, objects

    def init_scene(self):
        self.skybox = SkyBoxModel(self.app)
        self.main_light = DirectionalLight(self.app, position=(15, 25, 0), direction=(-15, -25, 0))

        add = self.add_model

        self.offset_x = len(self.map_[0])
        self.offset_z = len(self.map_)
        for z, line in enumerate(self.map_):
            for x, type_ in enumerate(line):
                if type_ == '#':
                    add(CubeModel(self.app, 
                                  position=((2 * x) - self.offset_x, -2, (2 * z) - self.offset_z), 
                                  texture_name='metal2', roughness=0.5, fresnel=0.8))

        objects = {}
        for (x, z), type_ in self.objects.items():
            if type_ == 'p': 
                self.player = Player(self.app, (2 * x) - self.offset_x, -0.6, (2 * z) - self.offset_z, self.map_, objects, self.offset_x, self.offset_z)
                add(self.player)
            if type_ == 'c':
                object_ = AnimatedCoin(self.app, (2 * x) - self.offset_x, -0.0, (2 * z) - self.offset_z)
                #add(object_)
                objects[(x, z)] = object_
            if type_ == 'a':
                object_ = AnimatedApple(self.app, (2 * x) - self.offset_x, -0.4, (2 * z) - self.offset_z)
                #add(object_)
                objects[(x, z)] = object_
            if type_ == 'g':
                position = ((2 * x) - self.offset_x + 1, -1, (2 * z) - self.offset_z - 1)
                object_ = Cat(self.app, position=position)
                self.lights.append(PointLight(self.app, position1=(position[0], 5, position[2]), position2=(0, 0, 0), colour=(1, 1, 1), attenuation=(1.5, 0.5, 0)))
                add(object_)
                #cube1 = StoneWall(self.app, position=((2 * x) - self.offset_x, -2, (2 * z) - self.offset_z))
                #add(cube1)
        self.objects = objects

    def render(self):
        #glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        #glBindFramebuffer(GL_FRAMEBUFFER, self.shadow_map.fbo)
        glViewport(0, 0, self.shadow_map.width, self.shadow_map.height)
        glBindFramebuffer(GL_FRAMEBUFFER, self.shadow_map.fbo)
        glClear(GL_DEPTH_BUFFER_BIT)
        glCullFace(GL_FRONT)
        for model in self.models:    
            if model.casts_shadow: 
                model.update_shadow()                 
                model.render_shadow()
        for model in self.objects.values():
            if model.casts_shadow and not model.collected: 
                model.update_shadow()                 
                model.render_shadow()
        glCullFace(GL_BACK)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(0, 0, self.app.width, self.app.height)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glDepthMask(GL_FALSE)
        
        if self.skybox is not None:
            self.skybox.update()   
            self.skybox.render()
        glDepthMask(GL_TRUE)
        for model in self.models:    
            model.update()   
            model.render()
        for model in self.objects.values():
            if not model.collected: 
                model.update()                 
                model.render()
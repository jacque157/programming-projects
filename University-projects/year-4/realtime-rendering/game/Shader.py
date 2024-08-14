from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader


MAX_POINT_LIGHTS = 10

class Shader:
    def __init__(self, vertex_path='shaders/default', fragment_path=None):
        self.program = self.compile_shaders(vertex_path, vertex_path if fragment_path is None else fragment_path)

    def compile_shaders(self, vertex_path, fragment_path):
        with open(f'{vertex_path}.vert', 'r') as file:
            vertex_text = file.readlines()

        with open(f'{fragment_path}.frag', 'r') as file:
            fragment_text = file.readlines()

        shader = compileProgram(
            compileShader(vertex_text, GL_VERTEX_SHADER),
            compileShader(fragment_text, GL_FRAGMENT_SHADER)
        )
        return shader
    
    def destroy(self):
        glDeleteProgram(self.program)

class DefaultShader(Shader):
    def __init__(self, vertex_path='shaders/default', fragment_path=None):
        super().__init__(vertex_path, fragment_path)
        self.uniforms_id = {'m_proj' : glGetUniformLocation(self.program, 'm_proj'),
                            'm_view' : glGetUniformLocation(self.program, 'm_view'),
                            'm_model' : glGetUniformLocation(self.program, 'm_model'),

                            'mainLight.position' : glGetUniformLocation(self.program, 'mainLight.position'),
                            'mainLight.direction' : glGetUniformLocation(self.program, 'mainLight.direction'),
                            'mainLight.colour' : glGetUniformLocation(self.program, 'mainLight.colour'),
                            
                            'material.Ia' : glGetUniformLocation(self.program, 'material.Ia'),
                            'material.Id' : glGetUniformLocation(self.program, 'material.Id'),
                            'material.Is' : glGetUniformLocation(self.program, 'material.Is'),
                            'material.shine' : glGetUniformLocation(self.program, 'material.shine'),
                            'material.roughness' : glGetUniformLocation(self.program, 'material.roughness'),
                            'material.F0' : glGetUniformLocation(self.program, 'material.F0'),
                            'shading' : glGetUniformLocation(self.program, 'shadingType'),
                            
                            'u_resolution' : glGetUniformLocation(self.program, 'u_resolution'),
                            'lightSpaceMatrix' : glGetUniformLocation(self.program, 'lightSpaceMatrix'),
                            'u_texture_0' : glGetUniformLocation(self.program, 'u_texture_0'),
                            'shadowMap' : glGetUniformLocation(self.program, 'shadowMap'),
                            
                            'pointLightsCount' : glGetUniformLocation(self.program, 'pointLightsCount'),
                            'toonShading' : glGetUniformLocation(self.program, 'toonShading')}
        
        for i in range(MAX_POINT_LIGHTS):
            self.uniforms_id[f'pointLights[{i}].position'] = glGetUniformLocation(self.program, f'pointLights[{i}].position')
            self.uniforms_id[f'pointLights[{i}].colour'] = glGetUniformLocation(self.program, f'pointLights[{i}].colour')
            self.uniforms_id[f'pointLights[{i}].attenuation'] = glGetUniformLocation(self.program, f'pointLights[{i}].attenuation')

    def set_matrix_f4x4(self, name, value, count=1):
        glUniformMatrix4fv(self.get_uniform_id(name), count, GL_TRUE, value)

    def set_vector_3f(self, name, value, count=1):
        glUniform3fv(self.get_uniform_id(name), count, value)

    def set_vector_2f(self, name, value, count=1):
        glUniform2fv(self.get_uniform_id(name), count, value)

    def set_float(self, name, value):
        glUniform1f(self.get_uniform_id(name), value)

    def set_int(self, name, value):
        glUniform1i(self.get_uniform_id(name), value)

    def set_bool(self, name, value):
        glUniform1i(self.get_uniform_id(name), value)

    def get_uniform_id(self, name):
        return self.uniforms_id[name]

class ShadowMap(Shader):
    def __init__(self, path='shaders/shadow_map'):
        super().__init__(path)
        self.uniforms_id = {'lightSpaceMatrix' : glGetUniformLocation(self.program, 'lightSpaceMatrix'),
                            'm_model' : glGetUniformLocation(self.program, 'm_model')}

    def set_matrix_f4x4(self, name, value):
        glUniformMatrix4fv(self.get_uniform_id(name), 1, GL_TRUE, value)

    def get_uniform_id(self, name):
        return self.uniforms_id[name]

class SkyBox(Shader):
    def __init__(self, path='shaders/skybox'):
        super().__init__(path)
        self.uniforms_id = {'m_proj' : glGetUniformLocation(self.program, 'm_proj'),
                            'm_view' : glGetUniformLocation(self.program, 'm_view'),
                            'm_model' : glGetUniformLocation(self.program, 'm_model'),
                            'skybox' : glGetUniformLocation(self.program, 'skybox'),
                            'toonShading' : glGetUniformLocation(self.program, 'toonShading')}

    def set_matrix_f4x4(self, name, value):
        glUniformMatrix4fv(self.get_uniform_id(name), 1, GL_TRUE, value)

    def set_bool(self, name, value):
        glUniform1i(self.get_uniform_id(name), value)

    def get_uniform_id(self, name):
        return self.uniforms_id[name]
    
class CookTorrance(DefaultShader):
    def __init__(self, vertex_path='shaders/cook_torrance', fragment_path='shaders/cook_torrance'):
        super().__init__(vertex_path, fragment_path)

class OrenNayar(DefaultShader):
    def __init__(self, vertex_path='shaders/oren_nayar', fragment_path='shaders/oren_nayar'):
        super().__init__(vertex_path, fragment_path)

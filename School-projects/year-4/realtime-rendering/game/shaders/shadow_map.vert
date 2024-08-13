#version 330 core

layout (location=0) in vec3 in_position;

uniform mat4 lightSpaceMatrix;
uniform mat4 m_model;

void main() {
    mat4 mvp = lightSpaceMatrix * m_model;
    gl_Position = mvp * vec4(in_position, 1.0);
}
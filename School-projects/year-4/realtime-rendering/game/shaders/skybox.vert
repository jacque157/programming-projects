#version 330 core
layout (location = 0) in vec3 in_position;

out vec3 TexCoords;

uniform mat4 m_proj;
uniform mat4 m_view;
uniform mat4 m_model; // identity

void main()
{
    TexCoords = in_position;
    gl_Position = m_proj * m_view * m_model * vec4(TexCoords, 1.0);
}  
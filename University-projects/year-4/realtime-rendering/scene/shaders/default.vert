#version 330 core

layout (location=0) in vec3 in_position;
layout (location=1) in vec3 in_normal;
layout (location=2) in vec2 in_textcoord;

out vec2 uv_0;
out vec3 normal;
out vec3 fragPos;
out vec4 shadowCoord;

uniform mat4 m_proj;
uniform mat4 m_view;
uniform mat4 lightSpaceMatrix;
uniform mat4 m_model;

void main()
{
    uv_0 = in_textcoord;
    fragPos = vec3(m_model * vec4(in_position, 1.0));
    normal = transpose(inverse(mat3(m_model))) * normalize(in_normal);
    gl_Position = m_proj * m_view * vec4(fragPos, 1.0);
    shadowCoord = lightSpaceMatrix *  vec4(fragPos, 1.0);
}
#version 330 core

const float NUMBER_OF_SHADES = 3;


out vec4 FragColor;

in vec3 TexCoords;

uniform samplerCube skybox;
uniform bool toonShading;

vec3 quantize(vec3 colour)
{
    float unit = 1.0 / NUMBER_OF_SHADES;
    vec3 unitsCount = floor(1 / unit * colour);
    return unitsCount * unit;
}

void main()
{    
    vec3 colour =  texture(skybox, TexCoords).rgb;
    FragColor = vec4(colour, 1.0);
}
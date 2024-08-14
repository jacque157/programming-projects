#version 330 core

const int MAX_POINT_LIGHTS = 10;
const float NUMBER_OF_SHADES = 4;
const float PI = 3.1415926538;

layout (location=0) out vec4 fragColour;

in vec2 uv_0;
in vec3 normal;
in vec3 fragPos;
in vec4 shadowCoord;

struct Material
{
    float Ia;
    float Id;
    float Is;
    float shine;
    float roughness;
    float F0;
};

struct PointLight
{   
    vec3 position;
    vec3 colour;
    vec3 attenuation;
};

struct DirectionalLight
{
    vec3 position;
    vec3 direction;
    vec3 colour;
};

uniform DirectionalLight mainLight;
uniform PointLight pointLights[MAX_POINT_LIGHTS];
uniform Material material;

uniform sampler2D u_texture_0;
uniform vec3 camPos;
uniform sampler2D shadowMap;
uniform vec2 u_resolution;
uniform int pointLightsCount;
uniform bool toonShading;

float quantize(float value, float number_of_units)
{
    float unit = 1.0 / number_of_units;
    float unitsCount = floor(value / unit);
    return unitsCount * unit;
}


float getShadow(vec3 direction) 
{
    vec3 projCoords = shadowCoord.xyz / shadowCoord.w;
    projCoords = projCoords * 0.5 + 0.5; 
    float closestDepth = texture(shadowMap, projCoords.xy).r;
    float currentDepth = projCoords.z;  
    vec3 lightDir = normalize(direction);
    
    float bias = max(0.05 * (1.0 - dot(normal, lightDir)), 0.005);  
    float shadow = currentDepth - bias > closestDepth  ? 1.0 : 0.0; 
    if(projCoords.z > 1.0)
        shadow = 0.0;
    return shadow;
}

float getSoftShadow(vec3 direction)
{
    vec3 projCoords = shadowCoord.xyz / shadowCoord.w;
    projCoords = projCoords * 0.5 + 0.5; 
    float currentDepth = projCoords.z; 
    vec3 lightDir = normalize(direction);
    float bias = max(0.05 * (1.0 - dot(normal, lightDir)), 0.005); 

    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(shadowMap, 0);
    for(float x = -1.5; x <= 1.5; x += 1)
        for(float y = -1.5; y <= 1.5; y += 1)
        {
            float pcfDepth = texture(shadowMap, projCoords.xy + vec2(x, y) * texelSize).r; 
            shadow += currentDepth - bias > pcfDepth ? 1.0 : 0.0; 
        }
    if(projCoords.z > 1.0)
        shadow = 0.0;
    return shadow /= 16.0;
}

float OrenNayarDiffuse(vec3 viewDir, vec3 lightDir, vec3 Normal)
{
    float NdotL = dot(Normal, lightDir);
    float NdotV = dot(Normal, viewDir);
    float NVAngle = acos(NdotL);
    float NLAngle = acos(NdotV);
    float alpha = max(NVAngle, NLAngle);
    float beta = min(NVAngle, NLAngle);
    float gama = dot(viewDir - (Normal * NdotV), lightDir - (Normal * NdotL));
    float mSquared = material.roughness * material.roughness;
    float A = 1 - (0.5 * mSquared / (mSquared + 0.57));
    float B = 0.45 * mSquared / (mSquared + 0.09);
    float C = sin(alpha) * tan(beta);
    float L1 = max(0, NdotL) * (A + (B * max(0, gama) * C));
    return L1;
}

float OrenNayarSpecular(vec3 viewDir, vec3 lightDir, vec3 Normal)
{
    return 0.0;
}

vec3 calcDirLight(vec3 colour, DirectionalLight light)
{
    vec3 Normal = normalize(normal);
    vec3 lightDir = normalize(-light.direction);
    vec3 viewDir = normalize(camPos - fragPos);
    vec3 diffuse = OrenNayarDiffuse(viewDir, -lightDir, Normal) * light.colour * material.Id;
    
    vec3 specular = OrenNayarSpecular(viewDir, -lightDir, Normal) * light.colour * material.Is;
    float shadow = getSoftShadow(lightDir);
    return colour * (((1.0-shadow) * dot(Normal, -lightDir) * (specular + diffuse)));
}

vec3 calcPointLight(vec3 colour, PointLight light)
{
    vec3 Normal = normalize(normal);
    vec3 lightDir = normalize(light.position - fragPos);
    vec3 viewDir = normalize(camPos - fragPos);
    vec3 diffuse = OrenNayarDiffuse(viewDir, -lightDir, Normal) * light.colour * material.Id;
    vec3 specular = OrenNayarSpecular(viewDir, -lightDir, Normal) * light.colour * material.Is;
    float distance = length(light.position - fragPos);
    float attenuation = light.attenuation.x + 
                        (light.attenuation.y * distance) +
                        (light.attenuation.z * distance * distance);
    return colour * dot(Normal, -lightDir) * (specular + diffuse) / attenuation;
}

vec3 getLight(vec3 colour)
{
    vec3 ambient = colour * material.Ia;
    vec3 finalcolour = ambient;
    finalcolour += calcDirLight(colour, mainLight);
    for (int i=0; i < MAX_POINT_LIGHTS && i < pointLightsCount; i++)
    {
        finalcolour += calcPointLight(colour, pointLights[i]);
    }
    return finalcolour;
}


void main()
{
    float gamma = 2.2;
    vec3 colour = texture(u_texture_0, uv_0).rgb;
    colour = pow(colour, vec3(gamma));
    colour = getLight(colour);
    colour = pow(colour, 1 / vec3(gamma));
    //vec3 colour = vec3(1.0, 0.0, 0.0);
    if (toonShading)
    {
        float luminance = dot(colour, vec3(0.2126, 0.7152, 0.0722));
        float gradient = fwidth(luminance );
        if (gradient > 0.3)
            colour = vec3(0.0);
    }
    fragColour = clamp(vec4(colour, 1.0), 0.0, 1.0);
}
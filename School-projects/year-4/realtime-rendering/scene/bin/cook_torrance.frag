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

float CookTorranceDiffuse(vec3 lightDir, vec3 Normal)
{
    return 1.0;
}

float CookTorranceSpecular(vec3 viewDir, vec3 lightDir, vec3 Normal)
{
    float spec = 0.0; 
    float diff = max(0.0, dot(lightDir, Normal));
    if (diff > 0.0)
    {
        float Ga = 1.0;
        vec3 half = normalize(viewDir + lightDir);
        float NdotH = dot(Normal, half);
        float NdotV = dot(Normal, viewDir);
        float VdotH = dot(viewDir, half);
        float NdotL = dot(Normal, lightDir);
        float mSquared = material.roughness * material.roughness;
        float LdotH = dot(lightDir, half);
        float Gb = 2 * NdotH * NdotV / VdotH;
        float Gc = 2 * NdotH * NdotL / LdotH;
        float G = min(Ga, min(Gb, Gc));
        float exponent = ((NdotH * NdotH) - 1) / (mSquared * NdotH * NdotH);
        float denominator = PI * mSquared * pow(NdotH, 4);
        float D = exp(exponent) / denominator;
        float theta = dot(half, viewDir);
        float F = material.F0 + ((1 - material.F0) * pow(1 - theta, 5));
        spec = F * G * D / (NdotV * NdotL * 4.0);
        if (toonShading)
            spec = quantize(spec, NUMBER_OF_SHADES - 1);
    }              
    return spec;
}

vec3 calcDirLight(vec3 colour, DirectionalLight light)
{
    vec3 Normal = normalize(normal);
    vec3 lightDir = normalize(-light.direction);
    vec3 diffuse = CookTorranceDiffuse(-lightDir, Normal) * light.colour * material.Id;
    vec3 viewDir = normalize(camPos - fragPos);
    vec3 specular = CookTorranceSpecular(viewDir, -lightDir, Normal) * light.colour * material.Is;
    float shadow = getSoftShadow(lightDir);
    return colour * (((1.0-shadow) * dot(Normal, -lightDir) * (specular + diffuse)));
}

vec3 calcPointLight(vec3 colour, PointLight light)
{
    vec3 Normal = normalize(normal);
    vec3 lightDir = normalize(light.position - fragPos);
    vec3 diffuse = CookTorranceDiffuse(-lightDir, Normal) * light.colour * material.Id;
    vec3 viewDir = normalize(camPos - fragPos);
    vec3 specular = CookTorranceSpecular(viewDir, -lightDir, Normal) * light.colour * material.Is;
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
    colour = vec3(1.0, 0.0, 0.0);
    fragColour = clamp(vec4(colour, 1.0), 0.0, 1.0);
}
#version 330 core

layout (location=0) out vec4 fragColour;

in vec2 uv_0;
in vec3 normal;
in vec3 fragPos;
in vec4 shadowCoord;

struct Light {
    bool pointLight;
    vec3 position;
    vec3 Ia;
    vec3 Id;
    vec3 Is;
};

uniform Light mainLight;
uniform sampler2D u_texture_0;
uniform vec3 camPos;
uniform sampler2D shadowMap;
uniform vec2 u_resolution;

float getShadow() 
{
    vec3 projCoords = shadowCoord.xyz / shadowCoord.w;
    projCoords = projCoords * 0.5 + 0.5; 
    float closestDepth = texture(shadowMap, projCoords.xy).r;
    float currentDepth = projCoords.z;  
    vec3 lightDir;
    if (mainLight.pointLight)
        lightDir = normalize(mainLight.position - fragPos);
    else
        lightDir = normalize(-mainLight.position);
    float bias = max(0.05 * (1.0 - dot(normal, lightDir)), 0.005);  
    float shadow = currentDepth - bias > closestDepth  ? 1.0 : 0.0; 
    if(projCoords.z > 1.0)
        shadow = 0.0;
    return shadow;
}

float getSoftShadow()
{
    vec3 projCoords = shadowCoord.xyz / shadowCoord.w;
    projCoords = projCoords * 0.5 + 0.5; 
    float currentDepth = projCoords.z; 
    vec3 lightDir;
    if (mainLight.pointLight)
        lightDir = normalize(mainLight.position - fragPos);
    else
        lightDir = normalize(-mainLight.position);
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

vec3 getLight(vec3 colour, Light light)
{
    vec3 Normal = normalize(normal);

    vec3 ambient = light.Ia;
    vec3 lightDir;
    if (light.pointLight)
        lightDir = normalize(light.position - fragPos);
    else
        lightDir = normalize(-light.position);
    float diff = max(0.0, dot(lightDir, Normal));
    vec3 diffuse = diff * light.Id;

    vec3 viewDir = normalize(camPos - fragPos);
    vec3 reflectDir = reflect(-lightDir, Normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = spec * light.Is;
    
    //float shadow = getShadow();
    float shadow = getSoftShadow();
    return colour * (((1.0-shadow) * (specular + diffuse)) + ambient);
    //return colour * (specular + diffuse + ambient);
}


void main()
{
    float gamma = 2.2;
    vec3 colour = texture(u_texture_0, uv_0).rgb;
    colour = pow(colour, vec3(gamma));
    colour = getLight(colour, mainLight);
    colour = pow(colour, 1 / vec3(gamma));
    //vec3 colour = vec3(1.0, 0.0, 0.0);
    fragColour = vec4(colour, 1.0);
}
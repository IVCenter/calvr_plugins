#version 460

out vec4 FragColor;

uniform vec4 diffuseVal;
uniform vec4 ambientVal;
uniform vec3 specularVal;
uniform float shininess;

in vs_out {
    vec3 FragPos;
    vec3 Normals;
    vec3 cameraPos;
} i;

uniform samplerCube skybox;

void main()
{
    float ratio = 1.00 / 2.42;
    vec3 I = normalize(i.FragPos - i.cameraPos);
    vec3 R = refract(I, normalize(i.Normals), ratio);
    FragColor = vec4(texture(skybox, R).rgb, 1.0);
}
#version 460

layout (location = 0) in vec3 vertex;
layout (location = 1) in vec3 normals;

out vs_out {
    vec3 FragPos;
    vec3 Normals;
    vec3 cameraPos;
} o;

uniform mat4 osg_ModelViewProjectionMatrix;
uniform mat4 osg_ModelViewMatrix;

uniform mat4 osg_ViewMatrix;
uniform mat4 osg_ViewMatrixInverse;
uniform mat4 osg_ProjectionMatrix;

void main() {
    o.Normals = normals;
    o.FragPos = vec3(osg_ViewMatrixInverse * osg_ModelViewMatrix * vec4(vertex, 1.0));
    o.cameraPos = osg_ViewMatrixInverse[3].xyz;

    gl_Position = osg_ModelViewProjectionMatrix * vec4(vertex, 1.0);
}
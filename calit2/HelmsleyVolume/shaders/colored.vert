#version 460

layout (location = 0) in vec3 vertex;
layout (location = 1) in vec3 normals;

out vs_out {
    vec3 FragPos;
    vec3 Normals;
    vec3 cameraPos;
    mat4 ModelMatrix;
} o;

uniform mat4 osg_ModelViewProjectionMatrix;
uniform mat4 osg_ModelViewMatrix;

uniform mat4 osg_ViewMatrix;
uniform mat4 osg_ViewMatrixInverse;
uniform mat4 osg_ProjectionMatrix;

void main() {
    o.Normals = mat3(osg_ViewMatrixInverse * osg_ModelViewMatrix) * normals;
    o.FragPos = vec3(osg_ViewMatrixInverse * osg_ModelViewMatrix * vec4(vertex, 1.0));
    o.cameraPos = osg_ViewMatrixInverse[3].xyz;
    o.ModelMatrix = osg_ViewMatrixInverse * osg_ModelViewMatrix;

    gl_Position = osg_ProjectionMatrix * osg_ViewMatrix * vec4(o.FragPos, 1.0);//vec4(vertex, 1.0);
}
#version 460

layout (location = 0) in vec3 vertex;
layout (location = 1) in vec2 uv;

out vs_out {
    vec2 uv;
} o;

uniform mat4 osg_ModelViewProjectionMatrix;

void main() {
    o.uv = uv;
    gl_Position = osg_ModelViewProjectionMatrix * vec4(vertex, 1.0);
}
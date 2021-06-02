#version 460

layout (location = 0) in vec3 vertex;
layout (location = 1) in vec2 uv;
layout (location = 2) in vec3 normal;

 	

out vs_out {
    vec2 uv;
    vec3 vertex;
    vec3 normal;
    vec3 fragment;
} o;

uniform mat4 osg_ModelViewProjectionMatrix;
uniform mat4 osg_ModelViewMatrix;
uniform mat4 osg_ViewMatrixInverse;

void main() {
	mat4 model = mat4(1.0);
    o.uv = uv;
    o.vertex = vertex;
    o.normal = normal;
    o.fragment = vec3(osg_ViewMatrixInverse * osg_ModelViewMatrix * vec4(vertex, 1.0));

    gl_Position = osg_ModelViewProjectionMatrix * vec4(vertex, 1.0);
}
#version 460

layout (location = 0) in vec3 vertex;

out vs_out {
	vec3 worldPosition;
} o;

uniform mat4 osg_ModelViewProjectionMatrix;
uniform mat4 osg_ModelViewMatrix;

uniform mat4 osg_ViewMatrixInverse;

void main() {
	vec4 sp = osg_ModelViewProjectionMatrix * vec4(vertex, 1.0);
	
	vec4 worldVert = osg_ViewMatrixInverse * osg_ModelViewMatrix * vec4(vertex, 1.0);
    o.worldPosition = worldVert.xyz;

	gl_Position = sp;
}
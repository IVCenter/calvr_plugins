#version 460

layout (location = 0) in vec3 vertex;

out vs_out {
	vec3 rd;
	vec3 sp;
} o;

uniform mat4 osg_ModelViewProjectionMatrix;
uniform mat4 osg_ModelViewMatrix;

uniform mat4 osg_ViewMatrix;
uniform mat4 osg_ViewMatrixInverse;
uniform mat4 osg_ModelViewMatrixInverse;
uniform mat4 osg_ProjectionMatrixInverse;

void main() {
	vec4 sp = osg_ModelViewProjectionMatrix * vec4(vertex, 1.0);
	
	vec4 worldVert = osg_ViewMatrixInverse * osg_ModelViewMatrix * vec4(vertex, 1.0);

	vec4 campos = osg_ViewMatrixInverse[3];

	mat4 world2obj = osg_ModelViewMatrixInverse * osg_ViewMatrix;

	o.rd = (world2obj * vec4(worldVert.xyz, 1)).xyz * 1000; //(worldVert.xyz - campos.xyz) / 100;
	o.sp = sp.xyw;

	gl_Position = sp;
}
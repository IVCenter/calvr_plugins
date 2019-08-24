#version 460

layout (location = 0) in vec3 vertex;

out vs_out {
	vec3 rd;
	vec3 ro;
	vec3 sp;
	mat4 WorldToObject;
} o;

uniform mat4 osg_ModelViewProjectionMatrix;
uniform mat4 osg_ModelViewMatrix;

uniform mat4 osg_ViewMatrix;
uniform mat4 osg_ViewMatrixInverse;
uniform mat4 osg_ProjectionMatrixInverse;

void main() {
	vec4 sp = osg_ModelViewProjectionMatrix * vec4(vertex, 1.0);
	
	vec4 worldVert = osg_ViewMatrixInverse * osg_ModelViewMatrix * vec4(vertex, 1.0);
	vec4 campos = osg_ViewMatrixInverse[3];
	vec3 dir = worldVert.xyz - campos.xyz;


	mat4 world2obj = inverse(osg_ModelViewMatrix) * osg_ViewMatrix;
	o.WorldToObject = world2obj;


	o.rd = ((world2obj * vec4(dir, 0)).xyz);
	o.ro = vec3(world2obj * vec4(campos.xyz, 1));
	o.sp = sp.xyw;

	gl_Position = sp;
}